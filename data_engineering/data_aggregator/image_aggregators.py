import os

import imageio
import numpy as np
from PIL import Image

from data_engineering.data_aggregator.parallelize import parallelize_task


def image_files_from_tif_to_npy(num_workers, npy_files_path, image_dir, image_prefixes):
    if not os.path.exists(npy_files_path):
        os.mkdir(npy_files_path)

    def image_to_npy(image_prefix):
        bands = [np.asarray(
            Image.open(f"{image_dir}/{image_prefix}/{image_prefix}_B{band}.tif"),
            dtype=np.uint16) for band in ["02", "03", "04"]]

        stacked_arr = np.stack(bands, axis=-1)
        np.save(f"{npy_files_path}/{image_prefix}", stacked_arr)

    def images_to_npy(image_prefixes):
        for image_prefix in image_prefixes:
            image_to_npy(image_prefix)

    parallelize_task(num_workers=num_workers, task=images_to_npy, iterator=image_prefixes)


def image_files_from_tif_to_normalized_and_augmented_png(num_workers, png_files_path, image_dir, image_prefixes,
                                                         image_suffix, image_stats,
                                                         augmentations):
    if not os.path.exists(png_files_path):
        os.mkdir(png_files_path)

    def image_to_png(image_prefix):
        bands = [np.asarray(
            Image.open(f"{image_dir}/{image_prefix}/{image_prefix}_B{band}.tif"),
            dtype=np.uint16) for band in ["02", "03", "04"]]

        stacked_arr = np.stack(bands, axis=-1)
        normalized_arr = (stacked_arr - image_stats['mean'].values) / image_stats['std'].values

        if augmentations is not None:
            augmented_arr = augmentations(image=normalized_arr)['image']
        else:
            augmented_arr = normalized_arr
        imageio.imwrite(im=augmented_arr, uri=f"{png_files_path}/{image_prefix}{image_suffix}.png",
                        format='PNG-FI')

    def images_to_png(image_prefixes):
        for image_prefix in image_prefixes:
            image_to_png(image_prefix)

    parallelize_task(num_workers=num_workers, task=images_to_png, iterator=image_prefixes)


def augmented_and_normalized_png_dataset(dataset: pd.DataFrame(), band_stats):
    flip = Compose([
        Flip(p=1)
    ])
    rotate = Compose([
        Rotate(limit=(0, 360), p=1),
    ])
    flip_and_rotate = Compose([
        Flip(p=1),
        Rotate(limit=(0, 360), p=1),
    ])

    datasets = []
    for augmentations, image_suffix in [(None, ''), (flip, '_flip'), (rotate, '_rotate'),
                                       (flip_and_rotate, '_flip_and_rotate')]:
        dataset_for_suffix = dataset.copy()
        dataset_for_suffix['image_prefix'] = dataset_for_suffix['image_prefix'] + image_suffix
        datasets.append(dataset_for_suffix)
        image_files_from_tif_to_normalized_and_augmented_png(
            num_workers=6, png_files_path=root + "/png_image_files/", image_dir=root + "/BigEarthNet-v1.0",
            image_prefixes=dataset['image_prefix'], image_stats=band_stats, image_suffix=image_suffix,
            augmentations=augmentations)

    return pd.concat(datasets)

train_with_augmentations = augmented_and_normalized_png_dataset(train, stats)