# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['extract_sea_ice']

import cv2
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
from rdp import rdp


def extract_sea_ice(img, kernel_size=9):

    if type(img) == str:
        img = cv2.imread(img, 0)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # blur image and apply Otsu`s binarization
    img_blur = cv2.medianBlur(img, 5)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    # denoise output with openings
    kernel = np.ones([kernel_size, kernel_size])
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # find centroids using distance transforms
    dist_transform = cv2.distanceTransform(img_thresh, cv2.DIST_FAIR, 0)
    _, centroid_masks = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    centroid_masks = centroid_masks.astype(np.uint8)

    # apply watershed segmentation
    background = cv2.dilate(img_thresh, kernel, iterations=1)
    unknown = cv2.subtract(background, centroid_masks)
    _, markers, _, centroids = cv2.connectedComponentsWithStats(centroid_masks, connectivity=8)
    markers = markers + 1
    markers[unknown == 255] = 0
    centroids_img = np.zeros(markers.shape)
    # print(centroids_img.shape)
    for cent in centroids:
         cent = [int(ele) for ele in cent]
         centroids_img[cent[1], cent[0]] = 255
    img_col = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    markers = cv2.watershed(img_col, markers)
    img_col[markers == -1] = [255, 255, 255]
    img_thresh = cv2.cvtColor(img_col, cv2.COLOR_RGB2GRAY).astype(np.uint8)

    # find blobs
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_thresh, connectivity=8)

    sizes = stats[:, -1]

    # minimum size of particles we want to keep (number of pixels)
    min_size = 650

    # empty output image
    img_thresh = np.zeros(output.shape, dtype=np.uint8)

    # for every component in the image, you keep it only if it's above min_size
    for i in range(1, nb_components):
        if sizes[i] >= min_size:
            img_thresh[output == i] = 255

    img_thresh[0, 0] = 0

    img_thresh = binary_fill_holes(img_thresh).astype(np.uint8) * 255
    return img_thresh
