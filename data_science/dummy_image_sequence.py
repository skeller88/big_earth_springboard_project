import time

import numpy as np

from data_science.augmented_image_sequence import AugmentedImageSequence


class DummyImageSequence(AugmentedImageSequence):
    def __getitem__(self, batch_num):
        if batch_num == 0 and self.has_verbose_logging:
            print('getting batch_num', batch_num)
            start = time.time()

        batch_y = self.y[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        batch_x = np.stack([np.full((120, 120, 3), fill_value=value) for value in batch_y])
        return batch_x, batch_y
