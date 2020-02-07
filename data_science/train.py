import datetime
import json
import os

from tensorflow.keras.models import load_model

from data_science.batch_generation_prototyping import random_seed


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


# def train_keras_model(*, x_train, y_train, x_valid, y_valid, bucket, model_dir, gcs_model_dir, gcs_log_dir,
#                       should_upload_to_gcs,
#                       experiment_name, start_model, should_train_from_scratch, should_regularize, lr, n_epochs=100,
#                       early_stopping_patience=6, metric_to_monitor='accuracy'):
#     model, model_base_metadata = get_model_and_metadata_from_gcs(bucket, model_dir, "h5", load_model, gcs_model_dir,
#                                                                  experiment_name)
#
#     model_and_metadata_filepath = os.path.join(model_dir, experiment_name)
#     gcs_model_and_metadata_filepath = os.path.join(gcs_model_dir, experiment_name)
#     gcs_log_dir = os.path.join(gcs_log_dir, experiment_name)
#
#     if model is None or should_train_from_scratch:
#         now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
#         model = start_model
#         model_base_metadata = {
#             'data': 'train_valid_google_automl_cloud_and_shadow_dataset_small.csv',
#             'data_prep': 'normalization_augmentation',
#             'experiment_name': experiment_name,
#             'experiment_start_time': now,
#             'model': 'keras_cnn',
#             'random_state': random_seed,
#             # so that initial_epoch is 0
#             'epoch': -1
#         }
#     else:
#         print('Resuming training at epoch', int(model_base_metadata['epoch']) + 1)
#
#     print(f'len(train): {len(x_train)}')
#
#     if x_valid is not None:
#         print(f'len(valid): {len(x_valid)}')
#
#     histories = []
#     metrics = ['accuracy']
#     loss = 'binary_crossentropy'
#
#     optimizer = Adam(learning_rate=lr)
#     model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
#
#     verbosity = 0
#     # Generators
#     #     train_generator = AugmentedImageSequenceFromNpy(x=x_train, y=y_train, batch_size=batch_size,
#     #                                                     augmentations=None, band_stats=stats)
#     #     valid_generator = AugmentedImageSequenceFromNpy(x=x_valid, y=y_valid, batch_size=batch_size,
#     #                                                     augmentations=None, band_stats=stats)
#     train_generator = get_image_dataset(x=x_train, y=y_train, augmentations=None, band_stats=stats,
#                                         batch_size=batch_size)
#     if should_regularize:
#         metric_to_monitor = f'val_{metric_to_monitor}'
#         valid_generator = get_image_dataset(x=x_valid, y=y_valid, augmentations=None, band_stats=stats,
#                                             batch_size=batch_size)
#     else:
#         valid_generator = None
#
#     callbacks = [
#         EarlyStopping(monitor=metric_to_monitor, patience=early_stopping_patience, verbose=verbosity),
#         ReduceLROnPlateau(monitor=metric_to_monitor, factor=0.5, patience=early_stopping_patience, min_lr=1e-6),
#         TensorBoard(gcs_log_dir, histogram_freq=1),
#     ]
#
#     if should_upload_to_gcs:
#         callbacks.append(
#             ModelCheckpointGCS(filepath=model_and_metadata_filepath, gcs_filepath=gcs_model_and_metadata_filepath,
#                                gcs_bucket=bucket, model_metadata=model_base_metadata, monitor=metric_to_monitor,
#                                verbose=verbosity))
#
#     history = model.fit_generator(train_generator, initial_epoch=int(model_base_metadata['epoch']) + 1,
#                                   epochs=n_epochs,
#                                   callbacks=callbacks,
#                                   validation_data=valid_generator,
#                                   shuffle=True, verbose=1)
#
#     actual_y_train, pred_y_train, pred_y_train_probs = get_predictions_for_dataset(train_generator, model)
#
#     if should_regularize:
#         actual_y_valid, pred_y_valid, pred_y_valid_probs = get_predictions_for_dataset(valid_generator, model)
#
#     if should_upload_to_gcs:
#         metadata_filepath = f"{model_and_metadata_filepath}_metadata.json"
#         with open(metadata_filepath, 'r') as json_file:
#             best_model_metadata = json.load(json_file)
#
#         best_model_metadata.update({
#             'accuracy_train': sklearn.metrics.accuracy_score(actual_y_train, pred_y_train),
#             'accuracy_valid': sklearn.metrics.accuracy_score(actual_y_valid, pred_y_valid),
#             'f1_score_train': sklearn.metrics.f1_score(actual_y_train, pred_y_train),
#             'f1_score_valid': sklearn.metrics.f1_score(actual_y_valid, pred_y_valid),
#             'confusion_matrix': numpy_to_json(sklearn.metrics.confusion_matrix(actual_y_valid, pred_y_valid)),
#             'precision_recall_curve': sklearn_precision_recall_curve_to_dict(
#                 sklearn.metrics.precision_recall_curve(actual_y_valid, pred_y_valid)),
#         })
#
#         with open(metadata_filepath, 'w+') as json_file:
#             json.dump(best_model_metadata, json_file)
#
#         blob = bucket.blob(f"{gcs_model_and_metadata_filepath}_metadata.json")
#         blob.upload_from_filename(metadata_filepath)
#
#     # Attempt to avoid memory leaks
#     del train_generator
#     del valid_generator
#     gc.collect()
#
#     if should_regularize:
#         return history, actual_y_valid, pred_y_valid, pred_y_valid_probs
#     else:
#         return history, actual_y_train, pred_y_train, pred_y_train_probs

    # if os.environ.get("SHOULD_PREDICT", "True") == "True":
#     pred_test_labels = predict(model=model, weight_dir=model_path, x=x_test, batch_size=batch_size, n_classes=n_classes)
#     clf_report = classification_report(y_test_labels, pred_test_labels, target_names=classes)
#     print(clf_report)