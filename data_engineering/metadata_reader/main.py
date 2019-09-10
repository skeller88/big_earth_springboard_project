import csv
import json
import os
import time

import tarfile

from google.cloud import storage

from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    gcs_client = storage.Client()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    metadata_blob_name: str = os.environ.get("GCS_METADATA_BLOB_NAME")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    uncompressed_blob_prefix: str = os.environ.get("UNCOMPRESSED_BLOB_PREFIX")
    disk_path: str = os.environ.get("DISK_PATH")

    tarfile_disk_path: str = disk_path + "/" + tarfile_blob_name

    if os.environ.get("SHOULD_DOWNLOAD_TARFILE") == "True":
        print(f"Downloading BigEarth tarfile from bucket {bucket_name} and blob {tarfile_blob_name}, saving to "
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

    if os.environ.get("SHOULD_EXTRACT_TARFILE") == "True":
        with tarfile.open(tarfile_disk_path, 'r') as fileobj:
            fileobj.extractall(path=disk_path)
        print(f"tar extracted from {tarfile_disk_path} to {extraction_path}")

    metadata_file = disk_path + "/" + "image_metadata.csv"
    bucket = gcs_client.bucket(bucket_name)

    num_files_uploaded = 0
    num_folders_uploaded = 0
    checkpoint = 1000
    start = time.time()
    with open(metadata_file, "w") as metadata_fileobj:
        writer = csv.DictWriter(metadata_fileobj,
                                fieldnames=["image_prefix", "labels", "coordinates", "projection", "tile_source",
                                            "acquisition_date"])

        # Don't use walk because filenames will have thousands of files. Iterate one by one instead
        for subdir_name in os.listdir(extraction_path):
            subdir_path = extraction_path + "/" + subdir_name
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if not filename.startswith("._"):
                        file_suffix = filename.rsplit(".")[-1]

                        content_type = "image/tiff" if file_suffix == ".tif" else "application/json"
                        blob_name: str = os.path.join(uncompressed_blob_prefix, subdir_name, filename)
                        print("blob_name", blob_name)
                        blob = bucket.blob(blob_name)

                        filepath = subdir_path + "/" + filename
                        blob.upload_from_filename(filepath, content_type=content_type)
                        num_files_uploaded += 1
                        if filename.endswith('.json'):
                            with open(filepath) as fileobj:
                                json_obj = json.loads(fileobj.read())
                                json_obj["image_prefix"] = subdir_name
                                writer.writerow(json_obj)
                num_folders_uploaded += 1

            if num_files_uploaded > checkpoint:
                elapsed = (time.time() - start) / 60
                print(f"Uploaded {num_files_uploaded} files and {num_folders_uploaded} folders in {elapsed} minutes, {num_files_uploaded / elapsed} files per minute.")
                checkpoint += 1000

    blob = bucket.blob(metadata_blob_name)
    blob.upload_from_filename(metadata_file)
    print(f"Uploaded metadata to {metadata_blob_name}")


if __name__ == "__main__":
    main()
