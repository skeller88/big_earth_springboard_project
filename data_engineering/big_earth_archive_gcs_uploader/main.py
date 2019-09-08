import os
import tarfile
from pathlib import Path

import pandas
from google.cloud import storage

from data_engineering.big_earth_archive_gcs_uploader.gcs_stream_uploader import GCSObjectStreamUploader

import requests
import shutil

gcs_client = storage.Client()
bucket_name: str = gcs_client.get_bucket(os.environ.get("GCS_BUCKET_NAME"))

with GCSObjectStreamUploader(client=gcs_client, bucket='test-bucket', blob='test-blob') as s:
    for _ in range(1024):
        s.write(b'x' * 1024)
    """
    Based on https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests/16696317#16696317
    :param url:
    :param local_fileobj:
    :return:
    """
    with requests.get(url, stream=True) as r:
        shutil.copyfileobj(r.raw, local_fileobj)
        local_fileobj