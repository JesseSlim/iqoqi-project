"""
Data fitting module for the IQOQI DIMM measurement setup.

Author: Jesse Slim, December 2016
"""

import numpy as np
from PIL import Image
import scipy.optimize
import scipy.signal
import ImageConversion
from enum import Enum


class FittingMethods(Enum):
    Disabled = 0
    EllipticalClampedGaussian = 1
    CircularClampedGaussian = 2
    Centroid = 3


def clamped_gaussian(height, center_x, center_y, width_x, width_y, theta, noiselevel, circular=False):
    """Returns a gaussian function, clamped to a maximum value of 1.0 to simulate saturation, with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    theta = float(theta)
    if circular:
        width_y = width_x
        theta = 0.0
    return lambda x,y: np.minimum(1.0, height*np.exp(
                -( ( ( np.cos(theta)*(center_x-x) - np.sin(theta)*(center_y-y) ) / width_x)**2
                  +( ( np.sin(theta)*(center_x-x) + np.cos(theta)*(center_y-y) ) / width_y)**2
                 ) / 2 ) + noiselevel)


def fit_clamped_gaussian(data, initial_params, circular=False):
    """Returns (height, x, y, width, noiselevel)
    the gaussian parameters of a 2D distribution found by a fit"""
    errorfunction = lambda p: np.ravel(clamped_gaussian(*p, circular=circular)(*np.indices(data.shape)) -
                                 data)
    try:
        if not circular:
            upper_bounds = [10.0, np.inf,  np.inf,  40.0, 40.0, 2*np.pi, 1.0]
            lower_bounds = [0.0,  -np.inf, -np.inf, 0.0,  0.0,  -2*np.pi,     0.0]
        else:
            upper_bounds = [10.0, np.inf, np.inf, 40.0, 0.0001, 0.0001, 1.0]
            lower_bounds = [0.0, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0]
        res = scipy.optimize.least_squares(errorfunction, initial_params, bounds=(lower_bounds, upper_bounds))
        return res.x
    except Exception as e:
        print("Fitting failed: " + repr(e))
        return np.array(initial_params)


def find_brightest_pixels_center(image):
    # if the image has color information, sum that out
    if image.ndim == 3:
        image = np.sum(image, axis=2)

    # convolve the image with a 3x3 kernel to find the center more reliably in weaker images
    conv_image = scipy.signal.convolve2d(image, np.ones((3, 3)), mode="same")
    conv_max_color_val = np.max(conv_image)
    conv_max_color_idx = np.flatnonzero(conv_image == conv_max_color_val)

    conv_max_color_positions = np.unravel_index(conv_max_color_idx, conv_image.shape)

    brightest_pixels_center = np.mean(conv_max_color_positions, axis=1)

    return brightest_pixels_center


def gaussian_fit_to_single_spot_image(image_flat, crop_rect, debug_images=False, circular=False):
    assert image_flat.ndim == 2

    max_color_idx = np.flatnonzero(image_flat == 255)
    num_saturated_pixels = len(max_color_idx)

    spot_center = find_brightest_pixels_center(image_flat)

    spot_crop_rect = np.zeros(4)
    spot_crop_rect[0:2] = spot_center + crop_rect[0:2]
    spot_crop_rect[2:4] = spot_center + crop_rect[2:4]

    crop_size = crop_rect[2:4] - crop_rect[0:2]

    x1, y1, x2, y2 = [(int(x) if int(x) > 0 else int(-x)) for x in spot_crop_rect]

    image_cropped = image_flat[x1:x2, y1:y2]
    image_cropped_linear = ImageConversion.uint8_color_img_to_linear_intensity_img(image_cropped)

    cropped_spot_center = spot_center - spot_crop_rect[0:2]
    initial_params = np.array([2.0, *cropped_spot_center, *(crop_size/4), 0.0, 0.0])
    if circular:
        initial_params[4] = 0.0
    fit_params = fit_clamped_gaussian(image_cropped_linear, initial_params, circular=circular)

    if debug_images:
        import matplotlib.pyplot as plt
        plt.imshow(image_cropped_linear, vmin=0.0, vmax=1.0)
        plt.show()
        plt.imshow(clamped_gaussian(*fit_params, circular=circular)(*np.indices(image_cropped_linear.shape)), vmin=0.0, vmax=1.0)
        plt.show()

    fit_params[1:3] += spot_crop_rect[0:2]

    if circular:
        relevant_params = [0, 1, 2, 3, 6] # width_y and theta are irrelevant
        all_fit_params = fit_params
        fit_params = all_fit_params[relevant_params]

    return np.array([*fit_params, num_saturated_pixels])


def centroid_fit_to_single_spot_image(image_flat, crop_rect, window_radius, debug_images=False):
    assert image_flat.ndim == 2

    max_color_idx = np.flatnonzero(image_flat == 255)
    num_saturated_pixels = len(max_color_idx)

    spot_center = find_brightest_pixels_center(image_flat)

    spot_crop_rect = np.zeros(4)
    spot_crop_rect[0:2] = spot_center + crop_rect[0:2]
    spot_crop_rect[2:4] = spot_center + crop_rect[2:4]

    crop_size = crop_rect[2:4] - crop_rect[0:2]

    x1, y1, x2, y2 = [(int(x) if int(x) > 0 else 0) for x in spot_crop_rect]

    image_cropped = image_flat[x1:x2, y1:y2]
    image_cropped_linear = ImageConversion.uint8_color_img_to_linear_intensity_img(image_cropped)

    xgrid, ygrid = np.indices(image_cropped.shape)
    mask = ((xgrid - spot_center[0] + spot_crop_rect[0])**2 + (ygrid - spot_center[1] + spot_crop_rect[1])**2) < window_radius**2
    windowed_image = np.where(mask, image_cropped_linear, 0)

    windowed_intensity = np.sum(windowed_image)

    centroid_x = np.sum(windowed_image*(xgrid))/windowed_intensity + spot_crop_rect[0]
    centroid_y = np.sum(windowed_image*(ygrid))/windowed_intensity + spot_crop_rect[1]

    if debug_images:
        import matplotlib.pyplot as plt
        plt.imshow(image_cropped_linear, vmin=0.0, vmax=1.0)
        plt.show()
        plt.imshow(windowed_image, vmin=0.0, vmax=1.0)
        plt.show()

    return np.array([windowed_intensity, centroid_x, centroid_y, num_saturated_pixels])


def fit_spots_from_image_file(filename, crop_size=None, displacements=None, debug_images=False, split_dim=None,
                              split_ratio=0.5, method=FittingMethods.Centroid, window_radius=5.0,
                              cpopt=ImageConversion.ColorProcessingOptions.cpMonochrome):
    full_image = Image.open(filename)
    full_imgarray = np.array(full_image)

    return fit_spots_from_array(full_imgarray, crop_size=crop_size, displacements=displacements,
                                debug_images=debug_images, split_dim=split_dim, split_ratio=split_ratio,
                                method=method, window_radius=window_radius, cpopt=cpopt)


def fit_spots_from_array(full_imgarray=None, spot_imgarrays=None, crop_size=None, displacements=None,
                         debug_images=False, split_dim=None, split_ratio=0.5,
                         method=FittingMethods.Centroid, window_radius=5.0, crop_offsets=None,
                         cpopt=ImageConversion.ColorProcessingOptions.cpMonochrome):
    if spot_imgarrays is None:
        total_size = np.shape(full_imgarray)[0:2]
        if split_dim is None:
            largest_dim = np.argmax(total_size)
            split_dim = largest_dim

        split_index = int(split_ratio * total_size[split_dim])

        spotA_imgarray, spotB_imgarray = np.split(full_imgarray, [split_index], axis=split_dim)
    else:
        spotA_imgarray, spotB_imgarray = spot_imgarrays

    # convert color images to monochrome images
    if cpopt == ImageConversion.ColorProcessingOptions.cpColor:
        cpopt = ImageConversion.ColorProcessingOptions.cpMonochrome

    spotA_imgarray = ImageConversion.apply_color_processing(spotA_imgarray, cpopt, flat=True)
    spotB_imgarray = ImageConversion.apply_color_processing(spotB_imgarray, cpopt, flat=True)

    if crop_size is None:
        crop_width = 40.0
        crop_height = 40.0
    else:
        crop_height, crop_width = crop_size

    if displacements is None:
        dpA = np.array([0, 0])
        dpB = np.array([0, 0])
    else:
        dpA, dpB = [np.array(x) for x in displacements]

    spotA_crop_rect = np.array([-crop_height/2, -crop_width/2, crop_height/2, crop_width/2]) + np.concatenate((dpA, dpA))
    spotB_crop_rect = np.array([-crop_height/2, -crop_width/2, crop_height/2, crop_width/2]) + np.concatenate((dpB, dpB))

    if method == FittingMethods.CircularClampedGaussian:
        fit_paramsA = gaussian_fit_to_single_spot_image(spotA_imgarray, spotA_crop_rect, debug_images=debug_images,
                                                        circular=True)
        fit_paramsB = gaussian_fit_to_single_spot_image(spotB_imgarray, spotB_crop_rect, debug_images=debug_images,
                                                        circular=True)
    elif method == FittingMethods.EllipticalClampedGaussian:
        fit_paramsA = gaussian_fit_to_single_spot_image(spotA_imgarray, spotA_crop_rect, debug_images=debug_images,
                                                        circular=False)
        fit_paramsB = gaussian_fit_to_single_spot_image(spotB_imgarray, spotB_crop_rect, debug_images=debug_images,
                                                        circular=False)
    elif method == FittingMethods.Centroid:
        fit_paramsA = centroid_fit_to_single_spot_image(spotA_imgarray, spotA_crop_rect, window_radius, debug_images=debug_images)
        fit_paramsB = centroid_fit_to_single_spot_image(spotB_imgarray, spotB_crop_rect, window_radius, debug_images=debug_images)
    else:
        raise ValueError("Invalid fitting method: " + str(method))

    if full_imgarray is not None:
        # offset the second spot fit position by width of the first spot to get absolute image coordinates again
        fit_paramsB[1 + split_dim] += split_index

    if crop_offsets is not None:
        fit_paramsA[1:3] += crop_offsets[0]
        fit_paramsB[1:3] += crop_offsets[1]

    return fit_paramsA, fit_paramsB


def get_fitparam_labels(method):
    if method == FittingMethods.EllipticalClampedGaussian:
        return ["ampl", "x", "y", "sigma_x", "sigma_y", "theta", "noise", "saturated_px"]
    elif method == FittingMethods.CircularClampedGaussian:
        return ["ampl", "x", "y", "sigma", "noise", "saturated_px"]
    elif method == FittingMethods.Centroid:
        return ["sum", "x", "y", "saturated_px"]
    else:
        raise ValueError("Invalid fitting method.")