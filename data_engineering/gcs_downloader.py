from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account


def main(bucket_name, blob_name):
    credentials = service_account.Credentials.from_service_account_file(
        "/Users/shanekeller/.gcs/big-earth-252219-fb2e5c109f78.json")
    gcs_client = storage.Client(credentials=credentials)
    _bucket = gcs_client.bucket(bucket_name)
    _blob = _bucket.blob(blob_name)

    with open(Path.home() / 'Documents/big_earth_springboard_project/test_archive_downloaded.tar.gz', 'wb') as fileobj:
        resp = _blob.download_as_string()
        fileobj.write(resp)


bucket_name: str = "big_earth"
blob_name: str = "test-blob"

main(bucket_name, blob_name)
