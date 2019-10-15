# %alias_magic t timeit
import cv2
import glob
import json
import os
from hashlib import sha256

import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

disk_dir = '/mnt/ssd-persistent-disk-200gb'
big_earth_dir = disk_dir + '/BigEarthNet-v1.0'
glob_path = big_earth_dir + '/**/*B0{}.tif'
# glob_path = big_earth_dir + '/S2B_MSIL2A_20180421T114349_42_65/*B0{}.tif'
os.listdir(disk_dir)

# %%t -n1 -r1
def imread_and_apply(filename, func):
    flattened = cv2.imread(filename, cv2.IMREAD_UNCHANGED).flatten()
    return func(flattened)

def apply_func_to_all_patches(per_image_func, da_gufunc):
    delayed_func = dask.delayed(per_image_func, pure=True)  # Lazy zversion of imread
    band_arrs = []
    for band_name, band_num in {"red": "2", "green": "3", "blue": "4"}.items():
        filenames = glob.glob(glob_path.format(band_num))
        lazy_images = [delayed_func(path) for path in filenames]   # Lazily evaluate imread on each path
        sample = lazy_images[0].compute()  # load the first image (assume rest are same shape/dtype)

        arrays = [da.from_delayed(lazy_image,           # Construct a small Dask array
                                  dtype=sample.dtype,   # for every lazy value
                                  shape=sample.shape)
                  for lazy_image in lazy_images]

        stack = da.stack(arrays, axis=0)                # Stack all small Dask arrays into one
        band_arrs.append(stack)
    bands = da.concatenate(band_arrs, axis=0)
    # https://docs.dask.org/en/latest/array-api.html#dask.array.gufunc.as_gufunc
    if da_gufunc is not None:
        return da.apply_gufunc(da_gufunc, "(i)->()", bands).compute()
    return bands.compute()

pixel_sum = apply_func_to_all_patches(lambda filename: imread_and_apply(filename, np.sum), np.sum)
print(pixel_sum)

def imread_and_validate_shape(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    assert 1 == (120, 120), f"filename shape was {img.shape}"
    return np.asarray(img.shape)

imread_and_validate_shape_delayed = dask.delayed(imread_and_validate_shape, pure=True)
bands = apply_func_to_all_patches(imread_and_validate_shape_delayed)

pixel_count = 120 * 120 * 3 * bands.shape[0]

mean = pixel_sum / pixel_count

def sum_mean_squared_diff(arr):
    squared_diffs = np.vectorize(lambda a, b: (a - b)**2)
    return np.sum(squared_diffs(arr, mean))

imread_and_get_mean_squared_diff = dask.delayed(lambda filename: imread_and_apply(filename, sum_mean_squared_diff), pure=True)
band_arrs = apply_func_to_all_patches(imread_and_get_mean_squared_diff, np.sum)
