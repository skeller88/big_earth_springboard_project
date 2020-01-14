# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from big_earth_springboard_project.data_science.image_loading import AugmentedImageSequenceFromDisk
!sudo pip3 install albumentations tensorflow-addons

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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


%matplotlib inline
%config InlineBackend.figure_format = 'retina'

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

dirname = 'gs://big_earth'
experiment_name = "basic_cnn_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

metadata_path = os.path.join(dirname, 'metadata')
train_path = os.path.join('raw_rgb/tiff')
test_path = train_path

weight_dir = os.path.join(dirname, 'model', 'model_weights')
log_dir = os.path.join(dirname, 'model', 'logs', experiment_name)

# fs = gcsfs.GCSFileSystem(project='big_earth', token='cloud')

for dir_to_create in [weight_dir, log_dir]:
    if not bucket.exists(dir_to_create):
        bucket.makedir(dir_to_create)

# ModelCheckpoint expects a path
weight_path = os.path.join(weight_dir, 'model_' + experiment_name + '_{epoch:02d}-{val_loss:.2f}.hdf5')

metadata_paths = bucket.ls("big_earth/metadata")

def load_metadata_from_gcs(filename):
    r = bucket.cat(filename)
    return pd.read_csv(io.BytesIO(r))

def load_image_bands_from_gcs(base_filename):
    bands = []
    for band in ["02", "03", "04"]:
        r = bucket.cat(base_filename.format(band))
        bands.append(imageio.core.asarray(imageio.imread(r, 'TIFF')))
    return np.stack(bands, axis=-1)

df = pd.concat(map(load_metadata_from_gcs, metadata_paths))
df = df.set_index('image_prefix', drop=False)

base_filename = os.path.join(train_path, df.index[0], df.index[0] + "_B{}.tif")
img = load_image_bands_from_gcs(base_filename)
print(img.shape)
print("stacked pixel for 3 bands\n", img[0][0])

print(len(df))
df.head()

df_no_cloud = df[
    (df['has_cloud_and_shadow'] == 0) &
    (df['has_snow'] == 0)
]
print('removed', len(df) - len(df_no_cloud), 'tiles with clouds.', len(df_no_cloud), 'tiles left')

non_label_columns = ['Unnamed: 0', 'image_prefix' ,'acquisition_date', 'coordinates', 'labels',
       'labels_sha256_hexdigest', 'projection', 'tile_source', 'has_snow', 'has_cloud_and_shadow']
df_labels = df_no_cloud.drop(columns=non_label_columns)
n_classes = len(df_labels.columns)
df_labels.head()

AUGMENTATIONS_TRAIN = Compose([
    Flip(p=0.5),
    Rotate(limit=(0, 360), p=0.5)
])

AUGMENTATIONS_TEST = Compose([])


def basic_cnn_model(img_shape, n_classes):
    """
    From https://arxiv.org/pdf/1902.06148.pdf

    To this end, we selected a shallow CNN architecture, which consists of three convolutional layers with 32, 32 and
    64 filters having 5 × 5, 5 × 5 and 3 × 3 filter sizes, respectively. We
    added one fully connected (FC) layer and one classification
    layer to the output of last convolutional layer. In all convolution operations, zero padding was used. We also applied
    max-pooling between layers.
    """
    img_inputs = Input(shape=img_shape)
    conv_1 = Conv2D(32, (5, 5), activation='relu')(img_inputs)
    maxpool_1 = MaxPooling2D((2, 2))(conv_1)
    conv_2 = Conv2D(32, (5, 5), activation='relu')(maxpool_1)
    maxpool_2 = MaxPooling2D((2, 2))(conv_2)
    conv_3 = Conv2D(64, (3, 3), activation='relu')(maxpool_2)
    flatten = Flatten()(conv_3)
    dense_1 = Dense(64, activation='relu')(flatten)
    output = Dense(n_classes, activation='sigmoid')(dense_1)

    return Model(inputs=img_inputs, outputs=output)


def pretrained_model(base_model_class, input_shape, output_shape):
    """
    All of the top performers use transfer learning and image augmentation: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/33559

    Another useful discussion on both topics: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36091#202629
    """
    # from https://www.kaggle.com/sashakorekov/end-to-end-resnet50-with-tta-lb-0-93#L321
    base_model = base_model_class(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    output = Dense(output_shape, activation='sigmoid')(x)
    model = Model(inputs=base_model.inputs, outputs=output)
    model.name = base_model.name

    return model


def train(model, model_preprocess_func, x_train: np.array, y_train: np.array, x_valid: np.array,
          y_valid: np.array, n_epochs, n_classes, batch_size, log_dir, weight_path, bucket):
    """
    Based on from https://www.kaggle.com/infinitewing/keras-solution-and-my-experience-0-92664
    """
    print(f'Split train: {len(x_train)}')
    print(f'Split valid: {len(x_valid)}')

    histories = []
    learn_rates = [0.001, 0.0001, 0.00001]
    metrics = [Accuracy(), Precision(), Recall(), F1Score(num_classes=n_classes, average='micro'),
               FBetaScore(num_classes=n_classes, beta=2.0, average='micro')]
    loss = 'binary_crossentropy'
    metric_to_monitor = 'val_loss'

    for learn_rate_num, learn_rate in enumerate(learn_rates):
        print(f'Training model on fold with learn_rate {learn_rate}')
        optimizer = SGD(lr=learn_rate, momentum=0.9)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        verbosity = 0
        callbacks = [
            EarlyStopping(monitor=metric_to_monitor, patience=2, verbose=verbosity),
            ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.5, patience=2, min_lr=0.000001),
            TensorBoard(log_dir, histogram_freq=1),
            ModelCheckpoint(weight_path, monitor=metric_to_monitor, save_weights_only=False, save_best_only=True,
                            verbose=verbosity)
        ]

        # Generators
        train_generator = AugmentedImageSequence(x=x_train, y=y_train, batch_size=batch_size,
                                                 model_preprocess_func=model_preprocess_func,
                                                 augmentations=AUGMENTATIONS_TRAIN, bucket=bucket)

        valid_generator = AugmentedImageSequence(x=x_valid, y=y_valid, batch_size=batch_size,
                                                 model_preprocess_func=model_preprocess_func,
                                                 augmentations=AUGMENTATIONS_TEST, bucket=bucket)

        history = model.fit_generator(generator=train_generator,
                                      epochs=n_epochs,
                                      steps_per_epoch=len(train_generator),
                                      callbacks=callbacks,
                                      validation_data=valid_generator, validation_steps=len(valid_generator),
                                      shuffle=True, verbose=1)
        histories.append(history)

    # data_valid will be shuffled at this point by the valid_generator, so create a new generator
    #     predict_generator = AugmentedImageSequence(x=x_valid, y=None, batch_size=valid_batch_size, model_preprocess_func=model_preprocess_func, augmentations=AUGMENTATIONS_TEST)
    #     pred_y_valid = model.predict_generator(predict_generator, steps=len(predict_generator))

    #     # Fine tune the class prediction thresholds by brute force on the validation data set
    #     # https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
    #     threshold = get_optimal_class_threshold(y_valid, pred_y_valid)

    #     # Store thresholds
    #     pd.DataFrame(data=threshold).transpose().to_csv(threshold_path, index=False)

    # Attempt to avoid memory leaks
    del train_generator
    del valid_generator
    #     del predict_generator
    gc.collect()

    return histories


def join_histories(histories):
    full_history = defaultdict(list)

    for history in histories:
        for key, value in history.history.items():
            full_history[key].extend(value)
    return full_history

def graph_model_history(history):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('Basic CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    max_epoch = len(history['val_loss'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(1, max_epoch, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")

n_epochs = 100
model = basic_cnn_model((120, 120, 3), n_classes=n_classes)
model_preprocess_func = None
fs = gcsfs.GCSFileSystem(project='big_earth', token='cloud')

x = df_labels['image_prefix'].values
# Example: gs://big_earth/raw_rgb/tiff/S2A_MSIL2A_20170613T101031_0_45/S2A_MSIL2A_20170613T101031_0_45_B02.tif
x = train_path + "/" + x + "/" + x + "_B{}.tif"

y = df_labels.drop(columns='image_prefix').values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.2)
# .8 * .25 = .2
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, test_size=.25)

# Test the correctness and speed of loading one batch
batch_size = 128
a = AugmentedImageSequenceFromDisk(x=xtrain[:batch_size], y=ytrain[:batch_size], batch_size=len(xtrain[:batch_size]),
                           augmentations=AUGMENTATIONS_TRAIN)

for x, y in a:
    print(x.shape, y.shape)
    break

a.on_epoch_end()

histories = train(model, model_preprocess_func,
      x_train=xtrain,
      y_train=ytrain,
      x_valid=xvalid,
      y_valid=yvalid,
      n_epochs=n_epochs,
      n_classes=n_classes,
      batch_size=len(xtrain),
      log_dir=log_dir,
      weight_path=weight_path,
      bucket=bucket)
