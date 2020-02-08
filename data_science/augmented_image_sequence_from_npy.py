import numpy as np

from data_science.augmented_image_sequence import AugmentedImageSequence


class AugmentedImageSequenceFromNpy(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, has_verbose_logging=False):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations,
                         has_verbose_logging=has_verbose_logging, should_test_time_augment=False)


    def batch_loader(self, image_paths, should_augment) -> np.array:
        imgs = np.array([np.load(image_path) for image_path in image_paths])
        normalized_imgs = (imgs - self.means) / self.stds

        if should_augment:
            return np.stack([self.augmentations(image=x)["image"] for x in normalized_imgs], axis=0)
        return np.array([x for x in normalized_imgs])


