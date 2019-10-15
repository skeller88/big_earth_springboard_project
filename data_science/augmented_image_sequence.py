import random
import time

import cv2
from keras.utils import Sequence
import numpy as np


from albumentations import (
    Compose, Flip, VerticalFlip, Resize, Rotate, ToFloat
)
import time

AUGMENTATIONS_TRAIN = Compose([
    Flip(p=0.5),
    Rotate(limit=(0, 360), p=0.5),
    Resize(width=256, height=256)
])

AUGMENTATIONS_TEST = Compose([
    Resize(width=256, height=256)
])


class AugmentedImageSequence(Sequence):
    def __init__(self, x: np.array, y: np.array, batch_size, model_preprocess_func, augmentations):
        self.x = x.copy()

        if y is not None:
            self.y = y.copy()
        else:
            self.y = y
        self.base_index = [idx for idx in range(len(x))]
        self.batch_size = batch_size
        self.model_preprocess_func = model_preprocess_func
        self.augmentations = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, batch_num):
        batch_x = self.x[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        if self.y is not None:
            batch_y = self.y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]

        start = time.time()
        images = [self.model_preprocess_func(image) for image in batch_x]

        if self.y is not None:
            batch_x = np.stack([self.augmentations(image=x)["image"] for x in images], axis=0)
            return batch_x, batch_y
        else:
            return np.array(images)

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

    def on_epoch_end(self):
        # https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        shuffled_index = self.base_index.copy()
        random.shuffle(shuffled_index)
        self.x = self.x[shuffled_index]
        self.y = self.y[shuffled_index]