import glob
import json
import os
from hashlib import sha256

import cv2
import dask.bag as db
import dask.dataframe as dd
import numpy as np

disk_dir = '/mnt/ssd-persistent-disk-200gb'
os.listdir(disk_dir)
big_earth_dir = disk_dir + '/BigEarthNet-v1.0'

# path = big_earth_dir + '/S2B_MSIL2A_20180421T114349_42_65/S2B_MSIL2A_20180421T114349_42_65_labels_metadata.json'
glob_path = big_earth_dir + '/**/*B0{}.tif'

# first, compute the average
pixel_mean = db.from_sequence(glob.glob(glob_path.format(band)) for band in ["2", "3", "4"]).mean()


def get_image_pixels(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return filename, img.flatten()


band_dfs_list = []
for band_name, band_num in {"red": "2", "green": "3", "blue": "4"}.items():
    band_bag = db.from_sequence(glob.glob(glob_path.format(band_num)))
    band_df = band_bag.to_dataframe(columns=['filename', '{}_pixels'.format(band_name)])
    band_dfs_list.append(band_df)

band_dfs = dd.dataframe.multi.concat(band_dfs_list)
band_dfs['pixels'].describe()

