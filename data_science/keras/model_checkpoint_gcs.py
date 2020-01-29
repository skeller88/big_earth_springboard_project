import os
import numpy as np
import warnings
from keras.callbacks import ModelCheckpoint
from tensorflow.python.lib.io import file_io


class ModelCheckpointGCS(ModelCheckpoint):
    """Taken from and modified:
    https://github.com/keras-team/keras/blob/tf-keras/keras/callbacks.py
    """
    def __init__(self, filepath, gcs_bucket, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointGCS, self).__init__(filepath=filepath, monitor=monitor, verbose=verbose,
                                                 save_best_only=save_best_only, save_weights_only=save_weights_only,
                                                 mode=mode, period=period)
        self.gcs_bucket = gcs_bucket

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, filepath))
                self.best = current
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)
                    with file_io.FileIO(filepath, mode='rb') as input_f:
                        with file_io.FileIO(filepath, mode='wb+') as output_f:
                            output_f.write(input_f.read())
            else:
                if self.verbose > 0:
                    print('Epoch %05d: %s did not improve' %
                          (epoch, self.monitor))
        else:
            if self.verbose > 0:
                print('Epoch %05d: saving model to %s' % (epoch, filepath))
            if self.save_weights_only:
                self.model.save_weights(filepath, overwrite=True)
            else:
                if is_development():
                    self.model.save(filepath, overwrite=True)
                else:
                    self.model.save(filepath.split(
                        "/")[-1])
                    with file_io.FileIO(filepath.split(
                            "/")[-1], mode='rb') as input_f:
                        with file_io.FileIO(filepath, mode='wb+') as output_f:
                            output_f.write(input_f.read())