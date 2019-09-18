import cv2
import numpy

base = "/Users/shanekeller/Documents/big_earth_springboard_project/big_earth/S2A_MSIL2A_20170717T113321_10_80/S2A_MSIL2A_20170717T113321_10_80_B{}.tif"

# bands = list(map(lambda band: Image.open(base.format(band)), ["02", "03", "04"]))
bands = list(map(lambda band: cv2.imread(base.format(band), cv2.IMREAD_UNCHANGED), ["02", "03", "04"]))
arr = numpy.stack(bands, axis=-1)
# https://stackoverflow.com/questions/2816144/python-convert-12-bit-image-encoded-in-a-string-to-8-bit-png
arr >>= 4
# arr = arr / 4095 * 255
# arr = arr.astype('uint8')
arr_num = arr[0][0][0]
# 120x120 pixels for 10m bands
print(arr.shape)
print(arr)
b = bands[0]
g = bands[1]
r = bands[2]

# https://stackoverflow.com/questions/25485886/how-to-convert-a-16-bit-to-an-8-bit-image-in-opencv
cv2.imwrite('output.jpg', arr)

img = cv2.imread('output.jpg', cv2.IMREAD_COLOR)
img_num = img[0][0][0]
print(img)

assert (arr == img).all()



# m = Image.merge("RGB", (b, g, r))
# marr = numpy.array(m)
# print(marr[0])