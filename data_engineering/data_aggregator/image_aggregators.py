# Run on machine if missing these libraries
# !pip3 install --upgrade scikit-learn
# !pip3 install --upgrade dask
# # This instance doesn't come with the necessary dependencies:
# # https://github.com/dask/distributed/issues/962
# !pip3 install dask[complete] distributed --upgrade

import glob
import json
import os
from hashlib import sha256

import dask.bag as db
import imageio
import numpy as np
import pandas as pd

import re


import dask
import dask.array as da
import dask.dataframe as dd
import dask.bag as db

import numpy as np
import time
import xarray as xr

def read_image(filename):
    return {
        'filename': filename,
        'image': np.random.randint(0, 4000, (3, 120, 120), dtype=np.uint16)
    }

n_images = 200
filenames = ["name_{}".format(name) for name in range(n_images)]

imgs = db.from_sequence(filenames).map(read_image).to_dataframe().set_index('filename').compute()



def image_files_from_tiff_to_npy(logger, tiff_dir, output_dir):
    def get_rgb_bands_from_filename_as_npy(filename):
        imgs = []
        for band in ["B02", "B03", "B04"]:
            image_filename = f"{filename}{filename.split('/')[-2]}_{band}.tif"
            imgs.append(imageio.core.asarray(imageio.imread(image_filename, 'TIFF')))
        return np.stack(imgs, axis=-1).flatten()

    paths = os.listdir(tiff_dir)
    logger.info(f"Fetched {len(paths)} paths.")
    paths_with_indexes = [(index, path) for index, path in enumerate(paths)]
    metadata = db.from_sequence(paths_with_indexes).map(read_and_augment_metadata)
    df = metadata.to_dataframe()
    df = df.set_index('index')

    # Check the dimensions
    logger.info(df.shape[0].compute())

    # 44 level 3 classes:
    # Currently using:
    # https://land.copernicus.eu/eagle/files/eagle-related-projects/pt_clc-conversion-to-fao-lccs3_dec2010
    # Should be using:
    # https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/
    clc = ["Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
           "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites", "Dump sites",
           "Construction sites", "Green urban areas", "Sport and leisure facilities", "Non-irrigated arable land",
           "Permanently irrigated land", "Rice fields", "Vineyards", "Fruit trees and berry plantations", "Olive groves",
           "Pastures", "Annual crops associated with permanent crops", "Complex cultivation patterns",
           "Land principally occupied by agriculture, with significant areas of natural vegetation", "Agro-forestry areas",
           "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grasslands", "Moors and heathland",
           "Sclerophyllous vegetation", "Transitional woodland-shrub", "Beaches, dunes, sands", "Bare rocks",
           "Sparsely vegetated areas", "Burnt areas", "Glaciers and perpetual snow", "Inland marshes", "Peat bogs",
           "Salt marshes", "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons", "Estuaries",
           "Sea and ocean"]

    for column in clc:
        df[column] = 0

    def multi_label_binarize_row(row):
        for label in row['labels']:
            row[label] = 1
        return row

    def multi_label_binarize_df(df):
        return df.apply(multi_label_binarize_row, axis=1)

    # Custom apply function uses GIL. Can't use Numba because there are python-specific objects in the function.
    # https://stackoverflow.com/questions/31361721/python-dask-dataframe-support-for-trivially-parallelizable-row-apply
    dfml = df.map_partitions(multi_label_binarize_df, meta=df.head(0)).compute(scheduler='processes')

    # Denote if patch has snow and/or clouds
    snow = pd.read_csv(os.path.join(cloud_and_snow_csv_dir, 'patches_with_seasonal_snow.csv'), header=None, names=['image_prefix'])
    snow_col = 'has_snow'
    snow[snow_col] = 1
    snow = snow.set_index('image_prefix')

    clouds = pd.read_csv(os.path.join(cloud_and_snow_csv_dir, 'patches_with_cloud_and_shadow.csv'), header=None, names=['image_prefix'])
    cloud_col = 'has_cloud_and_shadow'
    clouds[cloud_col] = 1
    clouds = clouds.set_index('image_prefix')

    print(snow.head(3))
    len_snow = len(snow)
    print('\n')
    print(clouds.head(3))
    len_clouds = len(clouds)

    for column in [snow_col, cloud_col]:
        dfml[column] = 0

    dfml = dfml.set_index('image_prefix')
    dfml.update(snow)
    dfml.update(clouds)
    assert dfml[snow_col].sum() == len_snow
    assert dfml[cloud_col].sum() == len_clouds

    # Atomically write to metadata_dir
    import shutil

    if os.path.exists(metadata_dir):
        shutil.rmtree(metadata_dir)

    os.mkdir(metadata_dir)

    dfml.to_csv(metadata_dir + '/metadata.csv')
