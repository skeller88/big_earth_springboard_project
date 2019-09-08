import os
import shutil
from io import BytesIO, BufferedIOBase
from pathlib import Path

import requests
from google.cloud import storage

from data_engineering.big_earth_archive_gcs_uploader.gcs_stream_uploader import GCSObjectStreamUploader

gcs_client = storage.Client()
bucket_name: str = os.environ.get("GCS_BUCKET_NAME")
blob_name: str = "test-blob"


    # with tarfile.open(Path.home() / 'Documents/big_earth_springboard_project/test_archive.tar.gz') as fileobj:
    #     file = fileobj.next()
    # s.write(data)
    # """
    # Based on https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests/16696317#16696317
    # :param url:
    # :param local_fileobj:
    # :return:
    # """
url = "http://bigearth.net/downloads/BigEarthNet-v1.0.tar.gz"

# with GCSObjectStreamUploader(client=gcs_client, bucket_name=bucket_name, blob_name=blob_name) as gcs_uploader:
#     with requests.get(url, stream=True) as response_stream:
#         for chunk in response_stream:
#             gcs_uploader.write(chunk)

with GCSObjectStreamUploader(client=gcs_client, bucket_name=bucket_name, blob_name=blob_name) as gcs_uploader:
    for line in open(Path.home() / 'Documents/big_earth_springboard_project/test_archive.tar.gz', 'rb'):
        print(line)
        gcs_uploader.write(line)
