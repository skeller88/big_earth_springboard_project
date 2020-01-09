from data_science.augmented_image_sequence import AugmentedImageSequence
!sudo
pip3
install
albumentations
tensorflow - addons

import io
from albumentations import (
    Compose, Flip, VerticalFlip, Resize, Rotate, ToFloat
)
import time

from concurrent import futures
from collections import Counter, defaultdict

import datetime

import os

print(os.listdir("."))
import threading

import gc
import gcsfs
from glob import glob
import imageio
from google.cloud import storage
from google.oauth2 import service_account

import numpy as np
import pandas as pd

import time

import scipy
from scipy.stats import bernoulli
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.metrics import Accuracy, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

from tensorflow_addons.metrics import FBetaScore, F1Score

from tqdm import tqdm

import random

import warnings

pal = sns.color_palette()
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf


random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
test_names = ['name_{}'.format(num) for num in range(128)]

import shutil

def write_test_data():
    if os.path.exists('../raw_rgb'):
        shutil.rmtree('../raw_rgb')

    os.mkdir('../raw_rgb')

    for test_name in test_names:
        dirname = '../raw_rgb/{}'.format(test_name)
        os.mkdir(dirname)
        for band in ["B02", "B03", "B04"]:
            imageio.imwrite(dirname + '/img_{}.tif'.format(band), np.random.randint(0, 4000, (120, 120)), 'tiff')

write_test_data()

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