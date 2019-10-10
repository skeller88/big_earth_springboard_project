# !pip install albumentations

from concurrent import futures
from collections import Counter, defaultdict

import os

import cv2
import gc
import bcolz
from glob import glob

import numpy as np
import pandas as pd

import keras as k
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History, TensorBoard
from keras.models import Model
from keras.layers import BatchNormalization, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence

import time

import scipy
from scipy.stats import bernoulli
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics import fbeta_score, precision_score, make_scorer, average_precision_score
from tqdm import tqdm

import random
import xgboost as xgb

from PIL import Image

import warnings

from data_science.augmented_image_sequence import AugmentedImageSequence

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

disk_dir = '/mnt/ssd-persistent-disk-200gb'
weight_dir = os.path.join(disk_dir, 'model_weights')
log_dir = os.path.join(disk_dir, 'logs')

for dir_to_create in [weight_dir, log_dir]:
    if not os.path.exists(dir_to_create):
        os.makedirs(dir_to_create)


nfolds = 5
kf = KFold(n_splits=nfolds, shuffle=False, random_state=random_seed)
x = sample_df['image_path'].values
y = labels
indexes = [(train_index, test_index) for train_index, test_index in kf.split(x)]


df_test['image_path'] = test_path + '/' + df_test['image_name'] + '.jpg'
avg_pred_test_probs, avg_thresholds = kfold_predict(model, model_preprocess_func=model_preprocess_func, weight_dir=weight_dir, x_test=df_test['image_path'].values, classes=classes, nfolds=nfolds, batch_size=batch_size)
