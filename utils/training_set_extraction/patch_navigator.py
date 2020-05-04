__all__ = ['PatchNavigator']

import os

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from .extract_sea_ice import extract_sea_ice


class PatchNavigator:
    def __init__(self, scenes_dir: str, patch_size: int, scn_res: tuple, out_dir: str,
                 masking_function=extract_sea_ice):
        # define attributes
        self.out_dir = out_dir
        self.mask_func = masking_function

        # read scenes directory
        self.scn_list = []
        for path, _, filenames in os.walk(scenes_dir):
            self.scn_list.extend([f"{path}/{file}" for file in filenames if file.endswith('.tif')])
        self.scn_idx = 0
        self.input_scn = None

        # patch_size and scene resolution
        self.patch_size = patch_size
        self.scn_res = scn_res
        self.scn_ratio = None

        # patch indexes
        self.i = -1
        self.j = 0
        self.max_i = None
        self.max_j = None

        # current scene, patch images and positive / negative masks
        self.curr_scn = None
        self.curr_patch = None
        self.curr_mask = None
        self.curr_outline = None
        self.pos_mask = np.ones([self.patch_size, self.patch_size], dtype=np.uint8) * 255
        self.neg_mask = np.zeros([self.patch_size, self.patch_size], dtype=np.uint8)

        # load_scene
        self.load_scene()

        # find maximum bounds
        self.find_bounds()

        # load patch
        self.next_cell()

        # create output dir
        for subdir in ['x', 'y_mask', 'y_outline']:
            os.makedirs(f"{self.out_dir}/{subdir}", exist_ok=True)

    def load_scene(self):
        self.input_scn = self.scn_list[self.scn_idx]
        with rasterio.open(self.input_scn) as src:
            self.curr_scn = src.read(1)
            self.scn_ratio = (self.curr_scn.shape[0] / self.scn_res[0],
                              self.curr_scn.shape[1] / self.scn_res[1])
            self.curr_scn = cv2.resize(self.curr_scn, dsize=self.scn_res)

    def load_patch(self):
        row_off = self.i * self.patch_size
        col_off = self.j * self.patch_size
        with rasterio.open(self.input_scn) as src:
            window = Window(col_off, row_off, self.patch_size, self.patch_size)
            self.curr_patch = src.read(window=window)[0, :, :]
            curr_patch_norm = cv2.normalize(self.curr_patch, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                            dtype=cv2.CV_8UC1)
            processed = self.mask_func(curr_patch_norm)
            self.curr_mask = processed['mask']
            self.curr_outline = processed['outline']

    def find_bounds(self):
        with rasterio.open(self.input_scn) as src:
            height, width = src.shape
            self.max_i = height // self.patch_size
            self.max_j = width // self.patch_size

    def next_cell(self):
        if self.i == self.max_i and self.j == self.max_j:
            print('Last cell, changing scene')
            self.next_scene()

        else:
            self.i += 1
            if self.i > self.max_i:
                self.i = 0
                self.j += 1

        try:
            self.load_patch()
        except:
            self.next_cell()

        if np.min(self.curr_patch) == 0:
            self.next_cell()

    def skip_cells(self):
        if self.i == self.max_i and self.j == self.max_j:
            print('Last cell, changing scene')
            self.next_scene()

        else:
            self.i += 2
            if self.i > self.max_i:
                self.i = 0
                self.j += 1

        self.load_patch()
        if np.min(self.curr_patch) == 0:
            self.next_cell()

    def next_col(self):
        if self.i == self.max_i and self.j == self.max_j:
            print('Last cell, changing scene')
        elif self.j == self.max_j:
            print('Last row, please advance cells instead')
        else:
            self.i = -1
            self.j += 1
            self.next_cell()

    def next_scene(self):
        if self.scn_idx == len(self.scn_list):
            print('last scene. closing app')
        else:
            self.scn_idx += 1
            self.i = -1
            self.j = 0
            self.load_scene()
            self.next_cell()

    def previous_scene(self):
        if self.scn_idx == 0:
            print('last scene. closing app')
        else:
            self.scn_idx -= 1
            self.i = -1
            self.j = 0
            self.load_scene()
            self.next_cell()

    def write_patch(self, label):
        fname = f"{os.path.basename(self.input_scn).split('.')[0]}_{self.i}_{self.j}.tif"
        cv2.imwrite(f"{self.out_dir}/x/{fname}", self.curr_patch)
        if label == 'keep':
            cv2.imwrite(f"{self.out_dir}/y_mask/{fname}", self.curr_mask)
            cv2.imwrite(f"{self.out_dir}/y_outline/{fname}", self.curr_outline)
        if label == 'positive':
            cv2.imwrite(f"{self.out_dir}/y_mask/{fname}", self.pos_mask)
            cv2.imwrite(f"{self.out_dir}/y_outline/{fname}", self.neg_mask)
        elif label == 'negative':
            cv2.imwrite(f"{self.out_dir}/y_mask/{fname}", self.neg_mask)
            cv2.imwrite(f"{self.out_dir}/y_outline/{fname}", self.neg_mask)
        self.next_cell()
