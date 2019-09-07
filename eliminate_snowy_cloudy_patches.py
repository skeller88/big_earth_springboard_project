#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This script eliminates patches with seasonal snow cover and cloud&shadow
# cover from the BigEarthNet Archive while reading the GeoTIFF files. It's based on
# https://gitlab.tubit.tu-berlin.de/rsim/bigearthnet-tools/blob/master/scripts/eliminate_snowy_cloudy_patches.py
# but actually copies patches that aren't snowy or cloudy to a dest_folder.
#
# Usage: eliminate_snowy_cloudy_patches.py [-h] [-r ROOT_FOLDER] [-e SNOW_FILE CLOUD_FILE]

from __future__ import print_function
import argparse
import os
import csv
import shutil

parser = argparse.ArgumentParser(
    description='This script eliminates patches with seasonal snow and cloud&shadow cover')
parser.add_argument('-r', '--root_folder', dest='root_folder',
                    help='root folder path contains multiple patch folders')
parser.add_argument('-d', '--dest_folder', dest='dest_folder',
                    help='destination folder')
parser.add_argument('-s', '--snow_file', dest='snow_file',
                    help='list of patches file for seasonal snow cover')
parser.add_argument('-c', '--cloud_file', dest='cloud_file',
                    help='list of patches file for cloud&shadow cover')
parser.add_argument('-e', '--snow_cloud_files', dest='snow_cloud_files', nargs='+',
                    help='list of patches files for seasonal snow and cloud&shadow cover')
args = parser.parse_args()

# Checks the existence of root folder of patches
if args.root_folder:
    if not os.path.exists(args.root_folder):
        print('ERROR: folder', args.root_folder, 'does not exist')
        exit()
else:
    print('ERROR: -r argument is required')
    exit()

# Checks the existence of csv files and populate the list of patches which will be eliminated
elimination_patch_list = []
for file_path in args.snow_cloud_files:
    if not os.path.exists(file_path):
        print('ERROR: file located at', file_path, 'does not exist')
        exit()
    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            elimination_patch_list.append(row[0])
print('INFO:', len(elimination_patch_list), 'number of patches will be eliminated')
elimination_patch_list = set(elimination_patch_list)

# Spectral band names to read related GeoTIFF files
band_names = ['B01', 'B02', 'B03', 'B04', 'B05',
              'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

patches_copied = 0
patches_skipped = 0
# Copies all patch directories except the ones in elimination list to the dest_folder
for root, dirs, files in os.walk(args.root_folder):
    if not root == args.root_folder:
        patch_folder_path = root
        patch_name = os.path.basename(patch_folder_path)
        dest_path = os.path.join(args.dest_folder, patch_name)
        if not patch_name in elimination_patch_list:
            print('Copying', patch_folder_path, 'to', dest_path)
            os.makedirs(dest_path)

            for filename in os.listdir(patch_folder_path):
                file_path = os.path.join(patch_folder_path, filename)
                shutil.copy2(file_path, dest_path)
            patches_copied += 1
        else:
            print('INFO: patch', patch_folder_path, 'was not copied')
            patches_skipped += 1


print(patches_copied, 'patches copied', patches_skipped, 'patches skipped')
