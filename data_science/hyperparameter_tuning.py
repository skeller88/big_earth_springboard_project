from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def optimize():
    def train_keras_with_hyperopt_params(params):
        experiment_name = (f"{project_name}_basic_cnn_lr_{params['learning_rate'][0]}_optimizer_"
                           f"{params['optimizer'][0]}_2020_2_08")
        result = train_keras_model(
            random_seed=random_seed, x_train=x_train[:10], y_train=y_train[:10], x_valid=x_valid, y_valid=y_valid,
            image_augmentations=augmentations_train, image_processor=None, band_stats=band_stats, bucket=bucket, model_dir=model_dir,
            gcs_model_dir=gcs_model_dir, gcs_log_dir=gcs_log_dir, experiment_name=experiment_name, start_model=model,
            should_train_from_scratch=True, optimizer=params['optimizer'][1], lr=params['learning_rate'][1],
            should_upload_to_gcs=True, n_epochs=2, early_stopping_patience=10)

        result['status'] = STATUS_OK
        return result

    space = {
        'learning_rate': hp.choice('learning_rate', [
            ('1e-4', 1e-4), ('1e-3', 1e-3), ('1e-2', 1e-2)]),
        'optimizer': hp.choice('optimizer', [
            ('Adam', Adam), ('SGD', SGD), ('RMSprop', RMSprop)])
    }
    trials = Trials()
    best = fmin(fn=train_keras_with_hyperopt_params,
                algo=tpe.suggest,
                space=space,
                max_evals=30,
                trials=trials)

    return best, trials

