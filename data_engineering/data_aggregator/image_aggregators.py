import os

import numpy as np
from PIL import Image

from data_engineering.data_aggregator.parallelize import parallelize_task


def image_files_from_tif_to_npy(npy_files_path, image_dir, image_prefixes):
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

    parallelize_task(num_workers=20, task=images_to_npy, iterator=image_prefixes)
