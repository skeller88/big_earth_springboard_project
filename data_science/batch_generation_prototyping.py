import os

from data_science.augmented_image_sequence import AugmentedImageSequence

print(os.listdir("."))

import imageio
from google.cloud import storage

import numpy as np
import pandas as pd

import time

import seaborn as sns
from sklearn.model_selection import train_test_split

import random

pal = sns.color_palette()

import tensorflow as tf


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
test_names = ['name_{}'.format(num) for num in range(128)]

import shutil

# From npy files
class AugmentedImageSequenceFromNpy(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations)

    def batch_loader(self, image_paths) -> np.array:
        return np.array([np.load(image_path) for image_path in image_paths])

metadata_file_path = f""
npy_files_path = f"{root}/data/big_earth/npy_files"
tiff_files_path = root + "/data/big_earth/BigEarthNet-V1.0"

xtrain_npy = (npy_files_path + "/" + sample.iloc[:5000]['image_prefix'] + ".npy").values
xtrain_tiff = (tiff_files_path + "/" + sample.iloc[:5000]['image_prefix'] + "/" +
               sample.iloc[:5000]['image_prefix'] + "_B{}.tif").values

ytrain = np.array([np.random.randn(1, 44) for _ in range(len(xtrain))])

batch_size = 128
np_sequence = AugmentedImageSequenceFromNpy(x=xtrain_npy, y=ytrain, batch_size=batch_size,
                                  augmentations=AUGMENTATIONS_TRAIN)


# From tiff on disk
class AugmentedImageSequenceFromTiff(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations)

    def batch_loader(self, image_paths) -> np.array:
        return np.array([self.load_image_bands_from_disk(image_path) for image_path in image_paths])

    def load_image_bands_from_disk(self, base_filename):
        bands = []
        for band in ["02", "03", "04"]:
            bands.append(np.array(Image.open(base_filename.format(band)), dtype=np.uint16))
        return np.stack(bands, axis=-1)

# Version 5 - write to zarr
def load_image_bands_from_disk(base_filename):
    bands = []
    for band in ["02", "03", "04"]:
        bands.append(imageio.core.asarray(imageio.imread(base_filename.format(band), 'TIFF')))
    return np.stack(bands, axis=-1)


def load_and_write_xarray_imgs(filenames):
    imgs = np.array([load_image_bands_from_disk(filename) for filename in filenames.values])
    arr = xr.DataArray(imgs, [('name', filenames.values),
                              ('x', [num for num in range(120)]),
                              ('y', [num for num in range(120)]),
                              ('bands', ['red', 'green', 'blue']),
                              ])

    filename = f'zarr_data/imgs_{filenames.index[0]}.zarr'
    ds = xr.Dataset({'data': arr})
    ds.to_zarr(filename)
    #     fs = gcsfs.GCSFileSystem(project='big_earth', token="/home/jovyan/work/.gcs/big-earth-252219-fb2e5c109f78.json")
    #     gcsmap = gcsfs.mapping.GCSMap('big_earth', gcs=fs, check=True, create=False)
    #     ds.to_zarr(store=gcsmap)
    return filename


start_time = time.time()
num_workers = 20
subset = sample.iloc[:30]['filepath']
chunk_size = len(subset) // num_workers
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    tasks = []
    start = 0
    for _ in range(num_workers):
        end = min(start + chunk_size, len(subset))
        tasks.append(executor.submit(load_and_write_xarray_imgs, subset[start:end]))
        start = end

filenames = [task.result() for task in tasks]
print(f'wrote files in {time.time() - start_time} seconds')


# Version 4 - load from bucket all at once
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/jovyan/work/.gcs/big-earth-252219-fb2e5c109f78.json"

from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor


def load_image_bands_from_gcs(filenames):
    credentials = service_account.Credentials.from_service_account_file(
        "/home/jovyan/work/.gcs/big-earth-252219-fb2e5c109f78.json")
    gcs_client = storage.Client(credentials=credentials)
    bucket = gcs_client.bucket("big_earth")

    imgs = []
    for filename in filenames:
        bands = []
        for band in ["02", "03", "04"]:
            blob = bucket.blob(filename.format(band))
            obj = blob.download_as_string()
            bands.append(imageio.core.asarray(imageio.imread(obj, 'TIFF')))
        imgs.append(np.stack(bands, axis=-1))

    return xr.DataArray(np.array(imgs), [('name', filenames),
                                         ('x', [num for num in range(120)]),
                                         ('y', [num for num in range(120)]),
                                         ('bands', ['red', 'green', 'blue']),
                                         ])


start_time = time.time()
num_workers = 20
subset = sample['gcs_base_path'].iloc[:10000]
chunk_size = len(subset) // num_workers
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    tasks = []
    start = 0
    for _ in range(num_workers):
        end = min(start + chunk_size, len(subset))

        tasks.append(executor.submit(load_image_bands_from_gcs, subset[start:end]))
        start = end
arr = xr.concat([task.result() for task in tasks], dim='name')

print(f'loaded data in {time.time() - start_time} seconds')


# Version 3 - load from numpy array
class AugmentedImageSequenceFromNpz(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, images: np.array):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations)
        self.images = images

    def batch_loader(self, image_paths) -> np.array:
        return self.images[image_paths]

def write_test_npy_data():
    def read_tile_data(filename):
        return np.random.randint(0, 4000, (3, 120, 120), dtype=np.uint16)

    n_images = 2000
    filenames = ["name_{}".format(name) for name in range(n_images)]

    obj = {filename: read_tile_data(filename) for filename in filenames}
    np.savez_compressed('train.npz', **obj)

write_test_npy_data()


start = time.time()
npz = np.load('../test.npz')
images = np.array(list(npz.values()))
image_names = np.array(list(npz.keys()))
labels = pd.DataFrame(data=[num for num in range(len(image_names))], index=image_names)

x = np.array([num for num in range(len(images))])
y = labels.loc[image_names].values

print('loaded images in', time.time() - start, 'seconds')

# Example: raw_rgb/tiff/S2A_MSIL2A_20170613T101031_0_45/S2A_MSIL2A_20170613T101031_0_45_B02.tif
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
# .8 * .25 = .2
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=.25)

# Basic performance testing that nothing goes wrong.
a = AugmentedImageSequenceFromNpz(x=xtrain[:batch_size], y=ytrain[:batch_size], batch_size=batch_size,
                                  augmentations=AUGMENTATIONS_TRAIN, images=images)

for x, y in a:
    print(x.shape, y.shape)
    break

a.on_epoch_end()

# Version 2 - load individual jpg band files from disk, convert to npy arrays, feed to generator
class AugmentedImageSequenceFromDisk(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations)

    def batch_loader(self, image_paths) -> np.array:
        return np.array([self.load_image_bands_from_gcs(image_path) for image_path in image_paths])

    def load_image_bands_from_disk(self, base_filename):
        bands = []
        for band in ["02", "03", "04"]:
            bands.append(imageio.core.asarray(imageio.imread(base_filename.format(band), 'TIFF')))
        return np.stack(bands, axis=-1)

def write_test_tiff_data():
    if os.path.exists('../raw_rgb'):
        shutil.rmtree('../raw_rgb')

    os.mkdir('../raw_rgb')

    for test_name in test_names:
        dirname = '../raw_rgb/{}'.format(test_name)
        os.mkdir(dirname)
        for band in ["B02", "B03", "B04"]:
            imageio.imwrite(dirname + '/img_{}.tif'.format(band), np.random.randint(0, 4000, (120, 120)), 'tiff')

write_test_tiff_data()

def get_x_and_y():
    img_paths = np.array([f"../test_img_data/{name}/{name}_{{}}.tif" for name in os.listdir('../test_img_data')])

    # increase the size of the test data to >128
    x = []
    for _ in range(100):
        x.extend(img_paths)
    x = np.array(x)
    y = np.array([num for num in range(len(x))])

    return x, y

# Version 1 - load individual jpg band files from google cloud storage, convert to npy arrays and stack in .npy files, feed to generator
class AugmentedImageSequenceFromGCS(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, bucket):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations)
        self.bucket = bucket

    def batch_loader(self, image_paths) -> np.array:
        return np.array([self.load_image_bands_from_gcs(image_path) for image_path in image_paths])

    def load_image_bands_from_gcs(self, base_filename):
        bands = []
        for band in ["02", "03", "04"]:
            blob = self.bucket.blob(base_filename.format(band))
            obj = blob.download_as_string()
            bands.append(imageio.core.asarray(imageio.imread(obj, 'TIFF')))
        return np.stack(bands, axis=-1)


def load():
    gcs_client = storage.Client()
    bucket = gcs_client.bucket("big_earth")

    # Generators
    train_generator = AugmentedImageSequenceFromGCS(x=x_train, y=y_train, batch_size=batch_size,
                                             augmentations=AUGMENTATIONS_TRAIN, bucket=bucket)

    valid_generator = AugmentedImageSequenceFromGCS(x=x_valid, y=y_valid, batch_size=batch_size,
                                             augmentations=AUGMENTATIONS_TEST, bucket=bucket)

    history = model.fit_generator(generator=train_generator,
                                  epochs=n_epochs,
                                  steps_per_epoch=len(train_generator),
                                  callbacks=callbacks,
                                  validation_data=valid_generator, validation_steps=len(valid_generator),
                                  shuffle=True, verbose=1)