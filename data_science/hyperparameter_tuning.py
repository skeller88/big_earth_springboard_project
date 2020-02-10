# Try hyperparameter optimization
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope

def optimize():
    def train_keras_with_hyperopt_params(params):
        model = basic_cnn_model_with_regularization((120, 120, 3), n_classes)
        experiment_name = (f"{project_name}_basic_cnn_early_stopping_patience_{params['early_stopping_patience']}_lr"
                           f"_{round(params['learning_rate'], 8)}_optimizer"
                           f"_{params['optimizer'][0]}_2020_2_08")
        print(experiment_name)
        return {
            'status': STATUS_OK,
            'loss': np.random.randn(1)[0]
        }
        result = train_keras_model(
            random_seed=random_seed, x_train=x_train[:10], y_train=y_train[:10], x_valid=x_valid, y_valid=y_valid,
            image_augmentations=augmentations_train, image_processor=None, band_stats=band_stats, bucket=bucket, model_dir=model_dir,
            gcs_model_dir=gcs_model_dir, gcs_log_dir=gcs_log_dir, experiment_name=experiment_name, start_model=model,
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