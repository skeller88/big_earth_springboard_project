import numpy as np

from data_science.augmented_image_sequence import AugmentedImageSequence


class AugmentedImageSequenceFromNpy(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, stats, has_verbose_logging=False):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations,
                         has_verbose_logging=has_verbose_logging)
        self.means = stats['mean'].values
        self.stds = stats['std'].values

    def batch_loader(self, image_paths) -> np.array:
        imgs = np.array([np.load(image_path) for image_path in image_paths])
        normalized_imgs = (imgs - self.means) / self.stds
        return normalized_imgs
