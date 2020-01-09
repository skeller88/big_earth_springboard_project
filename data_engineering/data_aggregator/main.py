import logging
import os
import sys
import tarfile
import time
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List

import imageio
from google.api_core.retry import Retry
from google.cloud import storage
import numpy as np

from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    """
    Downloads tarfile from $GCS_BUCKET_NAME/$GCS_TARFILE_BLOB_NAME, extracts tarfile to $DISK_PATH, and then
    traverses files in $DISK_PATH/$UNCOMPRESSED_BLOB_PREFIX. If $SHOULD_UPLOAD_AS_NPZ_FILE, writes npy file for each
    image patch, compresses to npz file, and uploads to gcs. Otherwise, uploads tiff files to gcs. In both cases,
    uploads metadata as json files to gcs.
    """
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    uncompressed_blob_prefix: str = os.environ.get("UNCOMPRESSED_BLOB_PREFIX")
    disk_path: str = os.environ.get("DISK_PATH")

    gcs_client = storage.Client()
    logger = logging.Logger("archive_etler_to_google_cloud", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    tarfile_disk_path: str = disk_path + "/" + tarfile_blob_name

    if os.environ.get("SHOULD_DOWNLOAD_TARFILE") == "True":
        logger.info(f"Downloading BigEarth tarfile from bucket {bucket_name} and blob {tarfile_blob_name}, saving to "
                    f"{tarfile_disk_path}")

        print("disk_path contents before download", os.listdir(disk_path))
        with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name,
                                       blob_name=tarfile_blob_name) as gcs_downloader:
            print("tarfile_disk_path", tarfile_disk_path)
            with open(tarfile_disk_path, 'wb') as fileobj:
                chunk = gcs_downloader.read()
                while chunk != b"":
                    fileobj.write(chunk)
                    chunk = gcs_downloader.read()

        print("disk_path contents after download", os.listdir(disk_path))

    extraction_path = tarfile_disk_path.replace(".gz", "").replace(".tar", "")
    logger.info(f"extraction_path: {extraction_path}")

    if os.environ.get("SHOULD_EXTRACT_TARFILE") == "True":
        with tarfile.open(tarfile_disk_path, 'r') as fileobj:
            fileobj.extractall(path=disk_path)
        logger.info(f"tar extracted from {tarfile_disk_path} to {extraction_path}")

    bucket = gcs_client.bucket(bucket_name)

    # Don't use walk because filenames will have thousands of files. Iterate one by one instead
    filepaths_to_upload = Queue()

    stats = {
        "num_files_uploaded": 0,
        "num_folders_uploaded": 0,
        "checkpoint": 1000
    }


def stack_image_bands_for_filename(base_filename):
    bands = []
    for band in ["02", "03", "04"]:
        bands.append(imageio.core.asarray(imageio.imread(base_filename.format(band), 'TIFF')))
    return np.stack(bands, axis=-1)



def upload_tiff_and_json_files(logger, filepaths_to_upload, bucket, stats, uncompressed_blob_prefix, extraction_path):
    google_retry = Retry(deadline=480, maximum=240)

    def on_google_retry_error(ex: Exception):
        logger.error("Exception when uploading blob to google cloud.")
        logger.exception(ex)

    def google_cloud_uploader():
        start = time.time()

        blob_name, filepath, content_type = filepaths_to_upload.get(timeout=30)
        while True:
            blob = bucket.blob(blob_name)
            try:
                google_retry(blob.upload_from_filename(filepath, content_type=content_type),
                             on_error=on_google_retry_error)
            except Exception as ex:
                logger.error(f"Uncaught exception when uploading blob to google cloud.")
                logger.exception(ex)
                filepaths_to_upload.put((blob.name, filepath, content_type))
                raise ex
            stats['num_files_uploaded'] += 1

            if stats['num_files_uploaded'] > stats['checkpoint']:
                elapsed = (time.time() - start) / 60
                logger.info(
                    f"Uploaded {stats['num_files_uploaded']} files in {elapsed} minutes, {stats['num_files_uploaded'] / elapsed} files per minute.")
                stats['checkpoint'] += 1000
            blob_name, filepath, content_type = filepaths_to_upload.get(timeout=5)

    def traverse_directory():
        for subdir_name in os.listdir(extraction_path):
            subdir_path = extraction_path + "/" + subdir_name
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if not filename.startswith("._"):
                        split = filename.rsplit(".")
                        if split[-1] == "tif" and split[-2].endswith(("B02", "B03", "B04")):
                            content_type = "image/tiff"
                            # multiple tiff files per subdirectory
                            blob_name: str = os.path.join(uncompressed_blob_prefix, "tiff", subdir_name, filename)

                            filepath = subdir_path + "/" + filename
                            filepaths_to_upload.put((blob_name, filepath, content_type))
                        elif split[-1] == "json":
                            # one json file per subdirectory
                            blob_name: str = os.path.join(uncompressed_blob_prefix, "json_metadata", filename)

                            filepath = subdir_path + "/" + filename
                            filepaths_to_upload.put((blob_name, filepath, content_type))

    num_workers = int(os.environ.get("NUM_WORKERS", 3))
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        tasks: List[Future] = []
        for x in range(num_workers):
            tasks.append(executor.submit(google_cloud_uploader))
        tasks.append(executor.submit(traverse_directory))
        logger.info(f"Started {len(tasks)} worker tasks.")

        logger.info("Starting traverse_directory")
        for task in as_completed(tasks):
            if task.exception() is not None:
                if type(task.exception()) == Empty:
                    logger.info("Child thread completed")
                else:
                    logger.error("Child thread failed")
                    logger.exception(task.exception())

    logger.info("Ending job")


if __name__ == "__main__":
    main()
