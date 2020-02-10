import datetime
import json
import os
from copy import copy

import sklearn
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import load_model

from data_science.keras.dataset import get_image_dataset, get_predictions_for_dataset
from data_science.keras.model_checkpoint_gcs import ModelCheckpointGCS
from data_science.serialization_utils import numpy_to_json, sklearn_precision_recall_curve_to_dict


def get_model_and_metadata_from_gcs(bucket, model_dir, model_file_ext, model_load_func, gcs_model_dir, experiment_name):
    model_and_metadata_filepath = os.path.join(model_dir, experiment_name)
    metadata_filepath = f"{model_and_metadata_filepath}_metadata.json"
    model_filepath = f"{model_and_metadata_filepath}.{model_file_ext}"

    gcs_model_and_metadata_filepath = os.path.join(gcs_model_dir, experiment_name)
    gcs_metadata_filepath = f"{gcs_model_and_metadata_filepath}_metadata.json"
    gcs_model_filepath = f"{gcs_model_and_metadata_filepath}.{model_file_ext}"

    gcs_metadata_blob = bucket.blob(gcs_metadata_filepath)
    gcs_model_blob = bucket.blob(gcs_model_filepath)

    if gcs_metadata_blob.exists():
        print('Downloading model blob.')
        gcs_metadata_blob.download_to_filename(metadata_filepath)

        with open(metadata_filepath, 'r') as json_file:
            model_metadata = json.load(json_file)

        model_metadata['epoch'] = int(model_metadata['epoch'])

        gcs_model_blob.download_to_filename(model_filepath)

        model = model_load_func(model_filepath)
        return model, model_metadata

    return None, None


def train_keras_model(*, random_seed, x_train, y_train, x_valid, y_valid, band_stats, image_augmentations,
                      image_processor, bucket, model_dir, gcs_model_dir, gcs_log_dir, should_upload_to_gcs,
                      experiment_name, model_name, start_model, should_train_from_scratch, optimizer, lr, batch_size=128,
                      n_epochs=100,
                      early_stopping_patience=6, metric_to_monitor='accuracy'):
    # TODO - deserialize existing model metadata from json
    model, model_base_metadata = get_model_and_metadata_from_gcs(bucket, model_dir, "h5", load_model, gcs_model_dir,
                                                                 experiment_name)

    model_and_metadata_filepath = os.path.join(model_dir, experiment_name)
    gcs_model_and_metadata_filepath = os.path.join(gcs_model_dir, experiment_name)
    gcs_log_dir = os.path.join(gcs_log_dir, experiment_name)

    if model is None or should_train_from_scratch:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        model = start_model
        model_base_metadata = {
            'data': 'train_valid_google_automl_cloud_and_shadow_dataset_small.csv',
            'data_prep': 'normalization_augmentation',
            'experiment_name': experiment_name,
            'experiment_start_time': now,
            'model': model_name,
            'random_state': random_seed,
            # so that initial_epoch is 0
            'epoch': -1,
            'optimizer': optimizer.__name__,
            'n_epochs': n_epochs,
            'early_stopping_patience': early_stopping_patience,
            'learning_rate': lr
        }
    else:
        print('Resuming training at epoch', int(model_base_metadata['epoch']) + 1)

    print(f'len(train): {len(x_train)}')

    if x_valid is not None:
        print(f'len(valid): {len(x_valid)}')
        metric_to_monitor = f'val_{metric_to_monitor}'
        valid_generator = get_image_dataset(x=x_valid, y=y_valid, augmentations=image_augmentations,
                                            image_processor=image_processor,
                                            band_stats=band_stats, batch_size=batch_size)
    else:
        valid_generator = None

    metrics = ['accuracy']
    loss = 'binary_crossentropy'

    optimizer = optimizer(learning_rate=lr)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    verbosity = 0
    train_generator = get_image_dataset(x=x_train, y=y_train, augmentations=image_augmentations,
                                        image_processor=image_processor,
                                        band_stats=band_stats, batch_size=batch_size)

    callbacks = [
        EarlyStopping(monitor=metric_to_monitor, patience=early_stopping_patience, verbose=verbosity),
        ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.5, patience=early_stopping_patience, min_lr=1e-6),
        TensorBoard(gcs_log_dir, histogram_freq=1),
    ]

    if should_upload_to_gcs:
        callbacks.append(
            ModelCheckpointGCS(filepath=model_and_metadata_filepath, gcs_filepath=gcs_model_and_metadata_filepath,
                               gcs_bucket=bucket, model_metadata=model_base_metadata, monitor=metric_to_monitor,
                               verbose=verbosity))

    history = model.fit_generator(train_generator, initial_epoch=int(model_base_metadata['epoch']) + 1,
                                  epochs=n_epochs,
                                  callbacks=callbacks,
                                  validation_data=valid_generator,
                                  shuffle=True, verbose=1)

    # load the best model
    if should_upload_to_gcs:
        best_model, best_model_metadata = get_model_and_metadata_from_gcs(bucket, model_dir, "h5", load_model,
                                                                          gcs_model_dir,
                                                                          experiment_name)
    else:
        best_model = model
        best_model_metadata = model_base_metadata

    y_actual_train, y_pred_train, y_pred_probs_train = get_predictions_for_dataset(train_generator, best_model)
    train_loss = binary_crossentropy(y_actual_train, y_pred_probs_train).numpy().tolist()

    # add more stats
    best_model_metadata.update({
        'history': history.history,
        'accuracy_train': sklearn.metrics.accuracy_score(y_actual_train, y_pred_train),
        'f1_score_train': sklearn.metrics.f1_score(y_actual_train, y_pred_train),
        'train_loss': train_loss,
        'loss': train_loss,
        'y_actual_train': y_actual_train,
        'y_pred_train': y_pred_train,
        'y_pred_probs_train': y_pred_probs_train,
    })

    if valid_generator is not None:
        y_actual_valid, y_pred_valid, y_pred_probs_valid = get_predictions_for_dataset(valid_generator, best_model)
        valid_loss = binary_crossentropy(y_actual_valid, y_pred_probs_valid).numpy().tolist()
        best_model_metadata.update({
            'accuracy_valid': sklearn.metrics.accuracy_score(y_actual_valid, y_pred_valid),
            'f1_score_valid': sklearn.metrics.f1_score(y_actual_valid, y_pred_valid),
            'confusion_matrix': sklearn.metrics.confusion_matrix(y_actual_valid, y_pred_valid),
            'precision_recall_curve': sklearn.metrics.precision_recall_curve(y_actual_valid, y_pred_valid),
            'y_actual_valid': y_actual_valid,
            'y_pred_valid': y_pred_valid,
            'y_pred_probs_valid': y_pred_probs_valid,
            'loss': valid_loss
        })

    if should_upload_to_gcs:
        serializable_metadata = copy(best_model_metadata)
        json_serializable_history = {}
        for k, v in history.history.items():
            json_serializable_history[k] = list(map(float, v))
        serializable_metadata['history'] = json_serializable_history

        if valid_generator is not None:
            serializable_metadata['confusion_matrix'] = numpy_to_json(serializable_metadata['confusion_matrix'])
            serializable_metadata['precision_recall_curve'] = sklearn_precision_recall_curve_to_dict(
                serializable_metadata['precision_recall_curve'])

        datasets = ["train", "valid"] if valid_generator is not None else ["train"]
        for dataset in datasets:
            for variable in ["y_actual", "y_pred", "y_pred_probs"]:
                serializable_metadata[f"{variable}_{dataset}"] = serializable_metadata[f"{variable}_{dataset}"].tolist()

        metadata_filepath = f"{model_and_metadata_filepath}_metadata.json"
        with open(metadata_filepath, 'w+') as json_file:
            json.dump(serializable_metadata, json_file)

        blob = bucket.blob(f"{gcs_model_and_metadata_filepath}_metadata.json")
        blob.upload_from_filename(metadata_filepath)

    return best_model_metadata
