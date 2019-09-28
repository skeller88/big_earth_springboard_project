import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def tiff_to_jpg(base_filename, optical_max_value: int):
    """
    Based on code in https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image/bigearthnet.py

    :param base_filename:
    :param optical_max_value: what to choose?
    https://community.hexagongeospatial.com/t5/ERDAS-IMAGINE/Rescale-from-12-bit-to-8-bit/m-p/4360/highlight/true
    :return:
    """
    # bands = list(map(lambda band: Image.open(base_filename.format(band)), ["02", "03", "04"]))
    bands = list(map(lambda band: cv2.imread(base_filename.format(band), cv2.IMREAD_UNCHANGED), ["02", "03", "04"]))
    rgb_img = np.stack(bands, axis=-1)
    # https://stackoverflow.com/questions/2816144/python-convert-12-bit-image-encoded-in-a-string-to-8-bit-png
    rgb_img = rgb_img / optical_max_value * 255.
    rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
    # 120x120 pixels for 10m bands
    print(rgb_img.shape)
    print("rgb_img pixel for 3 bands\n", rgb_img[0][0])

    # https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
    return rgb_img


def plot_versions(image_dir):
    image_base_name = image_dir.rsplit('/', 1)[1]
    image_base_filename = os.path.join(image_dir, f"{image_base_name}_B{{}}.tif")

    bands = list(map(lambda band: cv2.imread(image_base_filename.format(band), cv2.IMREAD_UNCHANGED), ["02", "03", "04"]))
    arr = np.stack(bands, axis=-1)
    # https://stackoverflow.com/questions/2816144/python-convert-12-bit-image-encoded-in-a-string-to-8-bit-png
    # arr = arr / 4095 * 255
    arr = arr.astype('uint8')
    b = bands[0]
    g = bands[1]
    r = bands[2]
    bgr = np.stack([b, g, r], axis=-1)
    bgr = (bgr / 2000 * 255.).astype(np.int8)
    print(bgr[0][0])

    # https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
    cv2.imwrite('output.jpg', bgr)

    img = cv2.imread('output.jpg')
    print(img[0][0])

    assert (bgr == img).all()
    return

    rgb_img_2000 = tiff_to_jpg(image_base_filename, optical_max_value=2000)
    rgb_img_4095 = tiff_to_jpg(image_base_filename, optical_max_value=4095)
    band_names = {"02": "blue", "03": "green", "04": "red"}
    bands = list(map(lambda band: (f"{band_names[band]}_tiff_band", Image.open(image_base_filename.format(band))),
                     ["02", "03", "04"]))

    metadata_file = os.path.join(image_dir, f"{image_base_name}_labels_metadata.json")
    with open(metadata_file) as metadata_file_obj:
        metadata = json.load(metadata_file_obj)

    fig = plt.figure(figsize=(15, 15))
    fig.suptitle(', '.join(metadata['labels']))
    rows = 2
    columns = 3
    imgs_to_plot = [("rgb_img_2000", rgb_img_2000), ("rgb_img_4095", rgb_img_4095)] + bands
    ax = []
    for i in range(1, columns * rows + 1):
        if i <= len(imgs_to_plot):
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(imgs_to_plot[i - 1][0])
            plt.imshow(imgs_to_plot[i - 1][1])

    plt.show()


# image_path = Path.home() / "data"
# images = os.listdir(image_path)

# pillow doesn't preserve pixel values

num_equal = 0
num_unequal = 0
img = Image.open('opencv_logo.jpg')
img.save('opencv_logo_saved.jpg')
imgr = Image.open('opencv_logo_saved.jpg')
for row in range(img.width):
    for column in range(img.height):
        try:
            assert (img.getpixel((row, column)) == imgr.getpixel((row, column)))
            num_equal += 1
        except Exception as ex:
            if num_unequal == 1:
                print('img', img.getpixel((row, column)))
                print('imgr', imgr.getpixel((row, column)))
            num_unequal += 1
print("num_equal", num_equal, "num_unequal", num_unequal)


# opencv-python doesn't preserve pixel values
img = cv2.imread('opencv_logo.jpg').astype(np.int8)
cv2.imwrite('opencv_logo_saved.jpg', img)
imgr = cv2.imread('opencv_logo_saved.jpg').astype(np.int8)
num_equal = 0
num_unequal = 0
for row in range(img.shape[0]):
    for column in range(img.shape[1]):
        try:
            assert (img[row][column] == imgr[row][column]).all()
            num_equal += 1
        except Exception as ex:
            if num_unequal == 1:
                print('img', img[row][column])
                print('imgr', imgr[row][column])
            num_unequal += 1
            # raise ex
print("num_equal", num_equal, "num_unequal", num_unequal)

# also tried reading with cvtColor
img = cv2.cvtColor(cv2.imread('opencv_logo.jpg'), cv2.COLOR_BGR2RGB).astype(np.int8)
cv2.imwrite('opencv_logo_saved.jpg', img)
imgr = cv2.cvtColor(cv2.imread('opencv_logo_saved.jpg'), cv2.COLOR_BGR2RGB).astype(np.int8)
num_equal = 0
num_unequal = 0
for row in range(img.shape[0]):
    for column in range(img.shape[1]):
        try:
            assert (img[row][column] == imgr[row][column]).all()
            num_equal += 1
        except Exception as ex:
            if num_unequal == 1:
                print('img', img[row][column])
                print('imgr', imgr[row][column])
            num_unequal += 1
            # raise ex

print("num_equal", num_equal, "num_unequal", num_unequal)

# plot_versions(os.path.join(image_path, images[0]))
# plot_versions(os.path.join(image_path, images[1]))
# plot_versions(os.path.join(image_path, images[2]))
