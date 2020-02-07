import numpy as np

from data_science.augmented_image_sequence import AugmentedImageSequence
import tensorflow


class AugmentedImageSequenceFromNpy(AugmentedImageSequence):
    def __init__(self, x: np.array, y: np.array, batch_size, augmentations, band_stats, has_verbose_logging=False):
        super().__init__(x=x, y=y, batch_size=batch_size, augmentations=augmentations,
                         has_verbose_logging=has_verbose_logging, should_test_time_augment=False)


    def batch_loader(self, image_paths, should_augment) -> np.array:
        imgs = np.array([np.load(image_path) for image_path in image_paths])
        normalized_imgs = (imgs - self.means) / self.stds

        if should_augment:
            return np.stack([self.augmentations(image=x)["image"] for x in normalized_imgs], axis=0)
        return np.array([x for x in normalized_imgs])


class ImageDataset(tensorflow.data.Dataset):
    def __new__(cls, image_paths, y, augmentations, band_stats):
        return tensorflow.data.Dataset.from_generator(
            cls._generator,
            output_types=tensorflow.dtypes.int64,
            output_shapes=((120, 120, 3),(1,)),
            args=(image_paths, y, augmentations, band_stats,)
        )

    def _generator(image_paths, y, augmentations, band_stats):
        means = band_stats['mean'].values
        stds = band_stats['std'].values
        for idx, image_path in enumerate(image_paths):
            img = np.load(image_path)
            normalized_img = (img - means) / stds

            if augmentations is not None:
                yield augmentations(normalized_img), y[idx]
            else:
                yield normalized_img, y[idx]

