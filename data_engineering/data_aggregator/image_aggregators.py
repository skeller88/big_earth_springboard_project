import imageio

import numpy as np


def image_files_from_tiff_to_npy(logger, image_dir, image_names, output_filepath):
    def get_rgb_bands_from_filename_as_npy(image_dir, image_name):
        imgs = []
        for band in ["B02", "B03", "B04"]:
            image_filename = f"{image_dir}/{image_name}{image_name}_{band}.tif"
            imgs.append(imageio.core.asarray(imageio.imread(image_filename, 'TIFF')))
        return np.stack(imgs, axis=-1).flatten()

    obj = {image_name: get_rgb_bands_from_filename_as_npy(image_dir, image_name) for image_name in image_names}

    np.savez_compressed(output_filepath, **obj)
