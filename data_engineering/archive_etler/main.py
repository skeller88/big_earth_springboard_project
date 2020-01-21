import logging
import os
import sys
import tarfile
import time
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue, Empty
from typing import List

from google.api_core.retry import Retry
from google.cloud import storage

from data_engineering.archive_etler.uploaders import upload_tiff_and_json_files
from data_engineering.data_aggregator.image_aggregators import image_files_from_tif_to_npy
from data_engineering.data_aggregator.metadata_aggregators import metadata_files_from_json_to_csv
from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    """
    Downloads tarfile from $GCS_BUCKET_NAME/$GCS_TARFILE_BLOB_NAME, extracts tarfile to $DISK_PATH, and then
    traverses files in $DISK_PATH/$UNCOMPRESSED_BLOB_PREFIX. If $SHOULD_UPLOAD_TIFF_AND_JSON_FILES,
    uploads tiff and json files to gcs.
    """
    global_start = time.time()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    uncompressed_blob_prefix: str = os.environ.get("UNCOMPRESSED_BLOB_PREFIX")
    should_upload_tiff_and_json_files: bool = os.environ.get("SHOULD_UPLOAD_TIFF_AND_JSON_FILES") == "True"
    disk_path: str = os.environ.get("DISK_PATH")

    gcs_client = storage.Client()
    logger = logging.Logger("archive_etler", level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)

    tarfile_disk_path: str = disk_path + "/" + tarfile_blob_name

    if os.environ.get("SHOULD_DOWNLOAD_TARFILE") == "True":
        start = time.time()
        logger.info(f"Downloading BigEarth tarfile from bucket {bucket_name} and blob {tarfile_blob_name}, saving to "
                    f"{tarfile_disk_path}")

        for blob_name in ["patches_with_cloud_and_shadow.csv",  "patches_with_seasonal_snow.csv"]:
            with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name, blob_name=blob_name) as gcs_downloader:
                with open(disk_path + "/" + blob_name, 'wb') as fileobj:
                    chunk = gcs_downloader.read()
                    while chunk != b"":
                        fileobj.write(chunk)
                        chunk = gcs_downloader.read()

        with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name,
                                       blob_name=tarfile_blob_name) as gcs_downloader:
            logger.info(f"tarfile_disk_path: {tarfile_disk_path}")
            with open(tarfile_disk_path, 'wb') as fileobj:
                chunk = gcs_downloader.read()
                while chunk != b"":
                    fileobj.write(chunk)
                    chunk = gcs_downloader.read()
        logger.info(
            f"Downloaded tarfile in {(time.time() - start) / 60} minutes.")

    extraction_path = tarfile_disk_path.replace(".gz", "").replace(".tar", "")
    logger.info(f"extraction_path: {extraction_path}")

    if os.environ.get("SHOULD_EXTRACT_TARFILE") == "True":
        start = time.time()
        with tarfile.open(tarfile_disk_path, 'r') as fileobj:
            fileobj.extractall(path=disk_path)
            # members = fileobj.getmembers()
            #
            # logger.info(f"Tarfile has {len(members)} files.")
            #
            # def extract(members):
            #     with tarfile.open(tarfile_disk_path, 'r') as fileobj:
            #         num_extracted = 0
            #         for member in members:
            #             fileobj.extract(member, path=disk_path)
            #             num_extracted += 1
            #             if num_extracted % 1e5 == 0:
            #                 logger.info(f"extracted {num_extracted} files.")
            #
            # num_workers = 5
            # chunk_size = len(members) // num_workers
            # with ThreadPoolExecutor(max_workers=num_workers) as executor:
            #     start_index = 0
            #     for _ in num_workers:
            #         end = min(start_index + chunk_size, len(members))
            #         print(start_index, end)
            #         executor.submit(extract, members[start_index:end])
            #         start_index = end
        # Remove the tarfile to save space
        os.remove(tarfile_disk_path)
        logger.info(
            f"tar extracted from {tarfile_disk_path} to {extraction_path} in {(time.time() - start) / 60} minutes.")

    if should_upload_tiff_and_json_files:
        bucket = gcs_client.bucket(bucket_name)
        # Don't use walk because filenames will have thousands of files. Iterate one by one instead
        filepaths_to_upload = Queue()

        stats = {
            "num_files_uploaded": 0,
            "num_folders_uploaded": 0,
            "checkpoint": 1000
        }

        upload_tiff_and_json_files(logger, filepaths_to_upload, bucket, stats, uncompressed_blob_prefix,
                                   extraction_path)
    start = time.time()
    metadata_df = metadata_files_from_json_to_csv(logger=logger, csv_files_path=disk_path + "/metadata",
                                                  cloud_and_snow_csv_dir=disk_path, json_dir=extraction_path)
    logger.info(f"Finished metadata aggregation in {(time.time() - start)} seconds.")

    start = time.time()

    logger.info(f"Starting image aggregation.")
    image_files_from_tif_to_npy(npy_files_path=disk_path + "/npy_image_files", image_dir=extraction_path,
                                image_prefixes=metadata_df['image_prefix'].values)
    logger.info(f"Finished image aggregation in {(time.time() - start)} seconds.")
    logger.info(f"Finished ETL in {(time.time() - global_start) / 60} minutes.")



if __name__ == "__main__":
    main()
