import logging
import os
import tarfile
import time
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List

from google.cloud import storage

from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    gcs_client = storage.Client()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    metadata_blob_name: str = os.environ.get("GCS_METADATA_BLOB_NAME")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    uncompressed_blob_prefix: str = os.environ.get("UNCOMPRESSED_BLOB_PREFIX")
    disk_path: str = os.environ.get("DISK_PATH")
    logger = logging.Logger("archive_etler_to_google_cloud")

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

    # with open(metadata_file, "w") as metadata_fileobj:
    # writer = csv.DictWriter(metadata_fileobj,
    #                         fieldnames=["image_prefix", "labels", "coordinates", "projection", "tile_source",
    #                                     "acquisition_date"])
    # if filename.endswith('.json'):
    #     with open(filepath) as fileobj:
    #         json_obj = json.loads(fileobj.read())
    #         json_obj["image_prefix"] = subdir_name
    #         writer.writerow(json_obj)

    # Don't use walk because filenames will have thousands of files. Iterate one by one instead

    filepaths_to_upload = Queue()

    stats = {
        "num_files_uploaded": 0,
        "num_folders_uploaded": 0,
        "checkpoint": 1000
    }

    def upload_with_retry(blob, content_type, filepath, retry_num):
        try:
            blob.upload_from_filename(filepath, content_type=content_type)
        except Exception as ex:
            logger.error(f"Exception when uploading to google cloud. Retry number {retry_num}")
            logger.exception(ex)
            if retry_num < 3:
                upload_with_retry(blob, content_type, filepath, retry_num + 1)

    def google_cloud_uploader():
        start = time.time()

        blob_name, filepath, content_type = filepaths_to_upload.get(timeout=30)
        while True:
            blob = bucket.blob(blob_name)
            upload_with_retry(blob, content_type, filepath, 0)
            stats['num_files_uploaded'] += 1

            if stats['num_files_uploaded'] > stats['checkpoint']:
                elapsed = (time.time() - start) / 60
                logger.info(f"Uploaded {stats['num_files_uploaded']} files in {elapsed} minutes, {stats['num_files_uploaded'] / elapsed} files per minute.")
                stats['checkpoint'] += 1000
            blob_name, filepath, content_type = filepaths_to_upload.get(timeout=5)

    def traverse_directory():
        for subdir_name in os.listdir(extraction_path):
            subdir_path = extraction_path + "/" + subdir_name
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if not filename.startswith("._"):
                        file_suffix = filename.rsplit(".")[-1]
                        content_type = "image/tiff" if file_suffix == ".tif" else "application/json"
                        blob_name: str = os.path.join(uncompressed_blob_prefix, subdir_name, filename)
                        filepath = subdir_path + "/" + filename
                        filepaths_to_upload.put((blob_name, filepath, content_type))

    num_workers = int(os.environ.get("NUM_WORKERS", 3))
    with ThreadPoolExecutor(max_workers=num_workers + 1) as executor:
        tasks: List[Future] = []
        for x in range(num_workers):
            logger.info(f"Starting worker {x}")
            tasks.append(executor.submit(google_cloud_uploader))
        tasks.append(executor.submit(traverse_directory))

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
