import csv
import os
import tarfile

from google.cloud import storage

from data_engineering.gcs_stream_downloader import GCSObjectStreamDownloader


def main():
    gcs_client = storage.Client()
    bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
    metadata_blob_name: str = os.environ.get("GCS_METADATA_BLOB_NAME")
    tarfile_blob_name: str = os.environ.get("GCS_TARFILE_BLOB_NAME")
    disk_path: str = os.environ.get("DISK_PATH")

    tarfile_disk_path: str = disk_path + "/" + tarfile_blob_name

    if os.environ.get("SHOULD_DOWNLOAD_TARFILE") == "True":
        print(f"Downloading BigEarth tarfile from bucket {bucket_name} and blob {tarfile_blob_name}, saving to "
              f"{tarfile_disk_path}")

        print("disk_path contents before download", os.listdir(disk_path))
        with GCSObjectStreamDownloader(client=gcs_client, bucket_name=bucket_name, blob_name=tarfile_blob_name) as gcs_downloader:
            print("tarfile_disk_path", tarfile_disk_path)
            with open(tarfile_disk_path, 'wb') as fileobj:
                chunk = gcs_downloader.read()
                while chunk != b"":
                    fileobj.write(chunk)
                    chunk = gcs_downloader.read()

        print("disk_path contents after download", os.listdir(disk_path))

    with tarfile.open(tarfile_disk_path, 'r') as fileobj:
        fileobj.extractall(path=disk_path)

    print(f"Extracted tar")

    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(metadata_blob_name)
    blob.upload_from_filename(disk_path + "/temp_names.csv")

if __name__ == "__main__":
    main()


