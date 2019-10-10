# Use on gcloud compute instance
# !pip3 install --upgrade scikit-learn
# !pip3 install --upgrade dask
# This instance doesn't come with the necessary dependencies:
# https://github.com/dask/distributed/issues/962
# !pip3 install dask[complete] distributed --upgrade

import cv2
import glob
import json
import os
from hashlib import sha256

import dask.bag as db
import dask.dataframe as dd

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

disk_dir = '/mnt/ssd-persistent-disk-200gb'
os.listdir(disk_dir)
big_earth_dir = disk_dir + '/BigEarthNet-v1.0'

path = big_earth_dir + '/S2B_MSIL2A_20180421T114349_42_65/S2B_MSIL2A_20180421T114349_42_65_labels_metadata.json'
# glob_path = big_earth_dir + '/S2B_MSIL2A_20180421T114349_42_65/*.json'
glob_path = big_earth_dir + '/**/*.json'

# labels
import re

replacements = {
    # '\'': '',
    # '[': '',
    # ']': '',
    'Bare rock': 'Bare rocks',
    'Natural grassland': 'Natural grasslands',
    'Peatbogs': 'Peat bogs',
    'Transitional woodland/shrub': 'Transitional woodland-shrub'
}

# Place longer ones first to keep shorter substrings from matching where the longer ones should take place
# For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
# 'hey ABC' and not 'hey ABc'
rep_sorted = sorted(replacements, key=len, reverse=True)

# Return string with all non-alphanumerics backslashed; this is useful if you want to match an arbitrary literal string that may have regular expression metacharacters in it.
rep_escaped = map(re.escape, rep_sorted)

# Create a big OR regex that matches any of the substrings to replace
pattern = re.compile("|".join(rep_escaped))


# For each match, look up the new string in the replacements, being the key the  old string
def multi_replace(string):
    return pattern.sub(lambda match: replacements[match.group(0)], string)


def multi_replace(arr):
    return [replacements[el] if replacements.get(el) is not None else el for el in arr]


# g['labels'] = g['labels'].apply(multi_replace).apply(lambda label: label.split(', '))

def read_and_augment_metadata(path_tuple):
    metadata_file = path_tuple[1]
    with open(metadata_file) as fileobj:
        obj = json.load(fileobj)
        obj['index'] = path_tuple[0]
        obj['labels_sha256_hexdigest'] = sha256('-'.join(obj['labels']).encode('utf-8')).hexdigest()
        obj['image_prefix'] = metadata_file.rsplit('/')[-2]
        return obj


paths = glob.glob(glob_path)
paths_with_indexes = [(index, path) for index, path in enumerate(paths)]
metadata = db.from_sequence(paths_with_indexes).map(read_and_augment_metadata)
df = metadata.to_dataframe()
# Dask doesn't have a reindex method
df = df.set_index('index')

# 44 level 3 classes:
# https://land.copernicus.eu/eagle/files/eagle-related-projects/pt_clc-conversion-to-fao-lccs3_dec2010
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
mlb = MultiLabelBinarizer(classes=clc)
encoded_labels = mlb.fit_transform(df['labels'])
ad = dd.from_array(encoded_labels, columns=mlb.classes)

print(ad.shape[0].compute())
ad.head(10)

# spot check that the indexes align
df.head(10)

ad_shape = ad.shape[0].compute()
df_shape = df.shape[0].compute()

print('ad', ad_shape)
print('df', df_shape)
assert ad_shape == df_shape

# Looks like Dask will keep the indexes aligned
j = df.join(ad)

df.to_csv(filename=big_earth_dir + '/metadata.csv', single_file=True)

dfr = dd.read_csv(big_earth_dir + '/metadata/metadata_*.csv', dtype={
    'Agro-forestry areas': 'float64', 'Airports': 'float64', 'Annual crops associated with permanent crops': 'float64',
    'Bare rocks': 'float64', 'Beaches, dunes, sands': 'float64', 'Broad-leaved forest': 'float64',
    'Burnt areas': 'float64', 'Coastal lagoons': 'float64', 'Complex cultivation patterns': 'float64',
    'Coniferous forest': 'float64', 'Construction sites': 'float64', 'Continuous urban fabric': 'float64',
    'Discontinuous urban fabric': 'float64', 'Dump sites': 'float64', 'Estuaries': 'float64',
    'Fruit trees and berry plantations': 'float64', 'Glaciers and perpetual snow': 'float64',
    'Green urban areas': 'float64', 'Industrial or commercial units': 'float64', 'Inland marshes': 'float64',
    'Intertidal flats': 'float64',
    'Land principally occupied by agriculture, with significant areas of natural vegetation': 'float64',
    'Mineral extraction sites': 'float64', 'Mixed forest': 'float64', 'Moors and heathland': 'float64',
    'Natural grasslands': 'float64', 'Non-irrigated arable land': 'float64', 'Olive groves': 'float64',
    'Pastures': 'float64', 'Peat bogs': 'float64', 'Permanently irrigated land': 'float64', 'Port areas': 'float64',
    'Rice fields': 'float64', 'Road and rail networks and associated land': 'float64', 'Salines': 'float64',
    'Salt marshes': 'float64', 'Sclerophyllous vegetation': 'float64', 'Sea and ocean': 'float64',
    'Sparsely vegetated areas': 'float64', 'Sport and leisure facilities': 'float64',
    'Transitional woodland-shrub': 'float64', 'Vineyards': 'float64', 'Water bodies': 'float64',
    'Water courses': 'float64'})
dfr.shape[0].compute()

g = dfr.groupby(by='labels_sha256_hexdigest').agg({'labels_sha256_hexdigest': 'count', 'labels': 'first'}).compute()
g.index.name = 'index'
g = g.rename({'labels_sha256_hexdigest': 'count'}, axis=1)

g.sort_values(by='count', ascending=False)[:20]
g['count'].describe()
np.log(g['count']).hist(bins=100)
