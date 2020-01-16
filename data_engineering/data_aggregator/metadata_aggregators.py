# Run on machine if missing these libraries
# !pip3 install --upgrade scikit-learn
# !pip3 install --upgrade dask
# # This instance doesn't come with the necessary dependencies:
# # https://github.com/dask/distributed/issues/962
# !pip3 install dask[complete] distributed --upgrade

import glob
import json
import os
import shutil
import time
from hashlib import sha256

import dask.bag as db

import pandas as pd


def metadata_files_from_json_to_csv(logger, cloud_and_snow_csv_dir, json_dir, csv_files_path):
    if not os.path.exists(csv_files_path):
        os.mkdir(csv_files_path)
    else:
        shutil.rmtree(csv_files_path)

    # From BigEarth team: we used the same labels of the CORINE Land Cover​ program operated by the European Environment
    # Agency. You can check the label names from
    # https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/.
    replacements = {
        'Bare rocks': 'Bare rock',
        'Natural grasslands': 'Natural grassland',
        'Peat bogs': 'Peatbogs',
        'Transitional woodland-shrub': 'Transitional woodland/shrub'
    }

    def multi_replace(arr):
        return [replacements[el] if replacements.get(el) is not None else el for el in arr]

    def read_and_augment_metadata(path_tuple):
        metadata_file = path_tuple[1]
        with open(metadata_file) as fileobj:
            obj = json.load(fileobj)
            # Dask doesn't have a reindex method, so build the index as files are being read
            obj['index'] = path_tuple[0]
            obj['labels'] = multi_replace(obj['labels'])
            obj['labels_sha256_hexdigest'] = sha256('-'.join(obj['labels']).encode('utf-8')).hexdigest()
            obj['image_prefix'] = metadata_file.rsplit('/')[-2]
            return obj

    glob_path = json_dir + '/**/*.json'
    paths = glob.glob(glob_path)
    logger.info(f"Fetched {len(paths)} paths.")
    start = time.time()
    paths_with_indexes = [(index, path) for index, path in enumerate(paths)]
    metadata = db.from_sequence(paths_with_indexes, npartitions=50).map(read_and_augment_metadata)
    df = metadata.to_dataframe()
    df = df.set_index('index')

    # Check the dimensions
    logger.info(df.shape[0].compute())
    logger.info(f"Read files into dataframe in {time.time() - start} seconds.")

    # 44 level 3 classes:
    # Currently using:
    # https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/
    clc = ["Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
           "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites",
           "Dump sites",
           "Construction sites", "Green urban areas", "Sport and leisure facilities", "Non-irrigated arable land",
           "Permanently irrigated land", "Rice fields", "Vineyards", "Fruit trees and berry plantations",
           "Olive groves",
           "Pastures", "Annual crops associated with permanent crops", "Complex cultivation patterns",
           "Land principally occupied by agriculture, with significant areas of natural vegetation",
           "Agro-forestry areas",
           "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland", "Moors and heathland",
           "Sclerophyllous vegetation", "Transitional woodland/shrub", "Beaches, dunes, sands", "Bare rock",
           "Sparsely vegetated areas", "Burnt areas", "Glaciers and perpetual snow", "Inland marshes", "Peatbogs",
           "Salt marshes", "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons",
           "Estuaries",
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

    # Denote if patch has snow and/or cloudsrandom_state
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

    dfml.to_csv(csv_files_path + '/metadata.csv')

    return dfml
