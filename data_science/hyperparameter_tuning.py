# Try hyperparameter optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import datetime
import json
import os
from collections import defaultdict
import gc
from joblib import dump, load
import random
import time
from typing import List, Tuple

from albumentations import (
    Compose, Flip, Rotate
)

import cv2
import dask
import dask.array as da

from google.cloud import storage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_score, precision_recall_curve

import tensorflow as tf
from tensorflow.keras.applications import inception_v3, resnet_v2
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import Accuracy

from data_engineering.dask_image_stats_collector import stats_for_numpy_images
from data_science.graph_utils import graph_model_history
from data_science.keras.model_checkpoint_gcs import ModelCheckpointGCS
from data_science.keras.cnn_models import basic_cnn_model, basic_cnn_model_with_regularization, pretrained_model
from data_science.serialization_utils import numpy_to_json, sklearn_precision_recall_curve_to_dict
from data_science.sklearn_batch_generator import SklearnBatchGenerator
from data_science.train import get_model_and_metadata_from_gcs, train_keras_model

root = '/home/jovyan/work'
random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/.gcs/big-earth-252219-fb2e5c109f78.json'
gcs_client = storage.Client()
bucket = gcs_client.bucket("big_earth")

n_classes = 1
n_epochs = 100
batch_size = 128

early_stopping_patience = 6
use_small_dataset = False
use_random_small_dataset = False

project_name = "cloud_and_shadow"
model_dir = os.path.join(root, "model/models")
log_dir = os.path.join(root, "model/logs")
# blob prefix
gcs_model_dir = "model/models"
# tensorboard
gcs_log_dir = "gs://big_earth/model/logs"

for directory in [log_dir, model_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def prepare_data(df):
    df['has_cloud_and_shadow_target'] = df['has_cloud_and_shadow_target'].apply(lambda x: np.array(json.loads(x)))
    df['binarized_labels'] = df['binarized_labels'].apply(lambda x: np.array(json.loads(x)))
    df['image_path'] = root + "/npy_image_files/" + df['image_prefix'] + ".npy"
    return df


df = pd.read_csv(root + "/metadata/metadata.csv")
df = prepare_data(df)
print(df['binarized_labels'].iloc[0].shape)
print(df['has_cloud_and_shadow_target'].iloc[0].shape)
df = df.set_index('image_prefix', drop=False)

google_automl_dataset = pd.read_csv( '/app/data_science/google_automl_cloud_and_shadow_dataset_small.csv')
google_automl_dataset['image_prefix'] = google_automl_dataset['gcs_uri'].str.split('/').apply(lambda x: x[-1].replace(".png", ""))
google_automl_dataset = google_automl_dataset.set_index('image_prefix', drop=False)

train = df.loc[google_automl_dataset[google_automl_dataset['set'] == 'TRAIN'].index]
valid = df.loc[google_automl_dataset[google_automl_dataset['set'] == 'VALIDATION'].index]
test = df.loc[google_automl_dataset[google_automl_dataset['set'] == 'TEST'].index]

print(len(train), len(valid), len(test))
print(len(train) + len(valid) + len(test) == len(google_automl_dataset))

stats_file = root + '/cloud_and_shadow_stats_all.csv'
if os.path.exists(stats_file):
    all_stats = pd.read_csv(stats_file)
else:
    stat_list = []
    npy_image_dir = root + "/npy_image_files"
    npy_files = [npy_image_dir + "/" + file + ".npy" for file in train['image_prefix'].values]
    start = time.time()
    stats = stats_for_numpy_images(npy_files, use_test_data=False)
    stats['data'] = 'all'
    stat_list.append(stats)

    # get stats per class
    no_cloud = train[train['has_cloud_and_shadow'] == 0]
    cloud = train[train['has_cloud_and_shadow'] == 1]
    print(len(no_cloud), len(cloud))

    for name, data in [('no_cloud', no_cloud), ('cloud', cloud)]:
        npy_files = [npy_image_dir + "/" + file + ".npy" for file in data['image_prefix'].values]
        stats = stats_for_numpy_images(npy_files, use_test_data=False)
        stats['data'] = name
        stat_list.append(stats)

    all_stats = pd.concat(stat_list)
    all_stats['band'] = all_stats.index
    all_stats = all_stats.reset_index()
    all_stats = all_stats.drop('index', axis=1)
    all_stats.to_csv(stats_file, index=False)

    print(f'stats computed in {time.time() - start}')

band_stats = all_stats[all_stats['data'] == 'all']

x_train = train['image_path'].values
x_valid = valid['image_path'].values
x_test = test['image_path'].values

target = 'has_cloud_and_shadow_target'
y_train = np.stack(train[target].values)
y_valid = np.stack(valid[target].values)
y_test = np.stack(test[target].values)

print(y_train.shape, y_train[0].shape)

augmentations_train = Compose([
    Flip(p=0.5),
    Rotate(limit=(0, 360), p=0.5),
])


def optimize():
    def train_keras_with_hyperopt_params(params):
        model = basic_cnn_model_with_regularization((120, 120, 3), n_classes)
        experiment_name = (f"{project_name}_basic_cnn_lr"
                           f"_{round(params['learning_rate'], 8)}_optimizer"
                           f"_{params['optimizer'][0]}_2020_2_08")
        print(experiment_name)

        result = train_keras_model(
            random_seed=random_seed, x_train=x_train[:10], y_train=y_train[:10], x_valid=x_valid, y_valid=y_valid,
            image_augmentations=augmentations_train, image_processor=None, band_stats=band_stats, bucket=bucket, model_dir=model_dir,
            gcs_model_dir=gcs_model_dir, gcs_log_dir=gcs_log_dir, experiment_name=experiment_name, start_model=model,
            model_name='keras_cnn',
            should_train_from_scratch=True, optimizer=params['optimizer'][1], lr=params['learning_rate'],
            should_upload_to_gcs=True, n_epochs=100, early_stopping_patience=params['early_stopping_patience'])

        result['status'] = STATUS_OK
        return result

    space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
        'early_stopping_patience': scope.int(hp.quniform('early_stopping_patience', 10, 30, 1)),
        'optimizer': hp.choice('optimizer', [
            ('Adam', Adam), ('SGD', SGD), ('RMSprop', RMSprop)])
    }
    trials = Trials()
    best = fmin(fn=train_keras_with_hyperopt_params,
                algo=tpe.suggest,
                space=space,
                max_evals=100,
                trials=trials)

    return best, trials