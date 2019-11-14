"""
Download some sample images from google cloud storage to a local directory, ~/data.
"""
import os
from pathlib import Path

from google.cloud import storage
from google.oauth2 import service_account
import imageio

credentials = service_account.Credentials.from_service_account_file(
    "/Users/shanekeller/.gcs/big-earth-252219-fb2e5c109f78.json")
gcs_client = storage.Client(credentials=credentials)
bucket = gcs_client.bucket("big_earth")
blob_name = "raw_test/S2A_MSIL2A_20170613T101031_0_45/raw_rgb_tiff_S2A_MSIL2A_20170613T101031_0_45_S2A_MSIL2A_20170613T101031_0_45_B02.tif"
blob = bucket.blob(blob_name)
obj = blob.download_as_string()
img = imageio.core.asarray(imageio.imread(obj, 'TIFF'))
print(obj)

# for image_name in ["S2A_MSIL2A_20170613T101031_34_81", "S2A_MSIL2A_20170613T101031_6_59",
#                    "S2A_MSIL2A_20170617T113321_52_67"]:
#     dirname = str(Path.home() / "data" / image_name)
#     os.makedirs(dirname, exist_ok=True)
#
#     for band in ["B02", "B03", "B04"]:
#         blob = bucket.blob(os.path.join("raw", "tiff", image_name, f"{image_name}_{band}.tif"))
#         blob.download_to_filename(os.path.join(dirname, f"{image_name}_{band}.tif"))
#
#     blob = bucket.blob(os.path.join("raw", "json", f"{image_name}_labels_metadata.json"))
#     blob.download_to_filename(os.path.join(dirname, f"{image_name}_labels_metadata.json"))


!pip install pympler

import dask.array as da
from distributed import Client
import numpy as np
import sys

client = Client()

def get_array(num_arrs):
    return da.from_array([np.random.randint(2000, size=(100, 100)) for num in range(num_arrs)], chunks=(num_arrs, 100, 100))

arr = get_array(1000)
print(sys.getsizeof(arr))
print(sys.getsizeof(arr.compute()))

# object memory usage
from pympler import muppy
from pympler import summary
import psutil
from distributed.utils import format_bytes

all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
summary.print_(sum1)

# process memory usage
proc = psutil.Process()
print(format_bytes(proc.memory_info().rss))