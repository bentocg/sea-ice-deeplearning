# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['extract_sea_ice']

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from rdp import rdp


def extract_sea_ice(img, outline=False, kernel_size=9):
    if type(img) == str:
        img = cv2.imread(img, 0)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if outline:
        img_blur = cv2.medianBlur(img, 5)
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2).astype(np.uint8)

    else:
        kernel = np.ones([kernel_size, kernel_size])
        img_blur = cv2.medianBlur(img, 5)
        _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

        img_thresh = cv2.erode(img_thresh, kernel)
        img_thresh = cv2.dilate(img_thresh, kernel)

        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity=8)

        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 350

        # your answer image
        img_thresh = np.zeros(output.shape)
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                img_thresh[output == i + 1] = 255

        img_thresh = binary_fill_holes(img_thresh).astype(np.uint8) * 255
        edges = cv2.findContours(img_thresh)
        if edges:
            edges = [rdp(ele) for ele in edges]
        img_thresh = np.zeros(output.shape)


    return img_thresh
