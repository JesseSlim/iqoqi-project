"""
Data processing module for the IQOQI DIMM measurement setup.

Author: Jesse Slim, December 2016
"""

import numpy as np
from enum import Enum
from PIL import Image

class ColorProcessingOptions(Enum):
    """
    Enumeration of possible color processing options
    """
    cpColor = 0
    cpMonochrome = 1
    cpRed = 2
    cpGreen = 3
    cpBlue = 4

def apply_color_processing(imgdata, cpopt, flat=False):
    """
    Apply a specific type of color processing to an RGB (NxMx3)-image

    Args:
        imgdata:    RGB image matrix
        cpopt:      color processing method to be applied

    Returns:
        RGB (NxMx3)-image with the selected color processing method applied
    """

    if cpopt == ColorProcessingOptions.cpColor:
        if flat:
            raise ValueError("Color images can not be represented as a flat image array without color information")
        return imgdata

    if cpopt == ColorProcessingOptions.cpRed:
        flat_imgdata = imgdata[:, :, 0]
    elif cpopt == ColorProcessingOptions.cpGreen:
        flat_imgdata = imgdata[:, :, 1]
    elif cpopt == ColorProcessingOptions.cpBlue:
        flat_imgdata = imgdata[:, :, 2]
    elif cpopt == ColorProcessingOptions.cpMonochrome:
        flat_imgdata = np.array(Image.fromarray(imgdata).convert("L"))

    if flat:
        return flat_imgdata

    color_index = {
        ColorProcessingOptions.cpRed: [0],
        ColorProcessingOptions.cpGreen: [1],
        ColorProcessingOptions.cpBlue: [2],
        ColorProcessingOptions.cpMonochrome: [0, 1, 2]
    }

    converted_imgdata = np.zeros_like(imgdata)
    for ci in color_index[cpopt]:
        converted_imgdata[:, :, ci] = flat_imgdata

    return converted_imgdata

def uint8_color_img_to_linear_intensity_img(imgarray):
    img_size = np.array(imgarray.shape[1::-1])  # invert the x and y-axis, images are saved in the opposite
                                                # column/row-order

    if imgarray.ndim == 2:
        imgarray = imgarray.reshape(img_size[1], img_size[0], 1)

    colors = imgarray.shape[2]

    imgarray_f = np.array(imgarray, dtype=np.float64) / 255.0
    # gamma decompression: https://en.wikipedia.org/wiki/Grayscale
    imgarray_linear = decompress_gamma(imgarray_f)

    if colors == 3:
        # grayscale luma conversion: https://en.wikipedia.org/wiki/Grayscale
        imgarray_1d = np.dot(imgarray_linear, np.array([0.2126, 0.7152, 0.0722]))
    else:
        imgarray_1d = imgarray_linear.reshape(img_size[1], img_size[0])

    return imgarray_1d

def decompress_gamma(data):
    data_linear = np.where(data > 0.04045, ((data + 0.055) / 1.055) ** 2.4, data / 12.92)

    return data_linear

def crop_image(imgarray, rect, return_crop_indices=True):
    x1, y1, x2, y2 = rect
    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 < 0: x2 = 0
    if y2 < 0: y2 = 0

    cropped_imgarray = imgarray[y1:y2, x1:x2, :]
    if return_crop_indices:
        return cropped_imgarray, x1, y1, x2, y2
    else:
        return cropped_imgarray