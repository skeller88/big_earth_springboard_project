import random
import time

import numpy as np
from tensorflow.keras.utils import Sequence


class AugmentedImageSequence(Sequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, has_verbose_logging):
        self.x = x
        self.y = y
        self.base_index = [idx for idx in range(len(x))]
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.has_verbose_logging = has_verbose_logging

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, batch_num):
        if batch_num == 0 and self.has_verbose_logging:
            print('getting batch_num', batch_num)
            start = time.time()

        batch_x = self.x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        if self.y is not None:
            batch_y = self.y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        start = time.time()
        images = self.batch_loader(batch_x)

        # training
        if self.y is not None:
            batch_x = np.stack([self.augmentations(image=x)["image"] for x in images], axis=0)

            if batch_num == 0 and self.has_verbose_logging:
                print('fetched batch_num', batch_num, 'in', time.time() - start, 'seconds')

            return batch_x, batch_y
        # test (inference only)
        else:
            return np.array(images)

    def batch_loader(self, image_paths) -> np.array:
        raise NotImplementedError()

    def on_epoch_end(self):
        # Can't figure out how to pass in the epoch variable
        #         print_with_stdout(f"epoch {epoch}, logs {logs}")
        #         if logs is not None:
        #             print_with_stdout(f"finished epoch {epoch} with accuracy {logs['acc']} and val_loss {logs['val_loss']}")
        #         else:
        #             print_with_stdout(f"finished epoch {epoch}")
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        shuffled_index = self.base_index.copy()
        random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]

        if self.y is not None:
            self.y = self.y[shuffled_index]
