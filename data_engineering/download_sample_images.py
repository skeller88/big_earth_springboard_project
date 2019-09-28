"""
Download some sample images from google cloud storage to a local directory, ~/data.
"""
import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    "/Users/shanekeller/.gcs/big-earth-252219-fb2e5c109f78.json")
gcs_client = storage.Client(credentials=credentials)
bucket = gcs_client.bucket("big_earth")

for image_name in ["S2A_MSIL2A_20170613T101031_34_81", "S2A_MSIL2A_20170613T101031_6_59",
                   "S2A_MSIL2A_20170617T113321_52_67"]:
    dirname = str(Path.home() / "data" / image_name)
    os.makedirs(dirname, exist_ok=True)

    for band in ["B02", "B03", "B04"]:
        blob = bucket.blob(os.path.join("raw", "tiff", image_name, f"{image_name}_{band}.tif"))
        blob.download_to_filename(os.path.join(dirname, f"{image_name}_{band}.tif"))

    blob = bucket.blob(os.path.join("raw", "json", f"{image_name}_labels_metadata.json"))
    blob.download_to_filename(os.path.join(dirname, f"{image_name}_labels_metadata.json"))
