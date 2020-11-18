import logging
import os
from glob import glob
from os import listdir
from os.path import splitext

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,  labels_file, dataset='training', augmentation=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        labels = pd.read_csv(labels_file)
        self.labels = [ele for ele in labels.loc[labels.dataset == dataset, 'label']]
        self.ids = [ele for ele in labels.loc[labels.dataset == dataset, 'image']]
        self.augmentation = augmentation
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def only_positive(self):
        self.ids = [ele for idx, ele in enumerate(self.ids) if self.labels[idx] == 'positive_easy']

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = img_nd.reshape((img_nd.shape[0], img_nd.shape[1], 1)).astype(np.uint8)

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx)
        img_file = glob(self.imgs_dir + idx)

        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        img = self.preprocess(Image.open(img_file[0]))
        if len(mask_file) == 1:
            mask = self.preprocess(Image.open(mask_file[0]))
        else:
            mask = np.zeros(img.shape)
        mask[mask < 200] = 0
        mask[mask > 0] = 1

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            return {'image': sample['image'].transpose(2, 0, 1).astype('float32'),
                    'mask': sample['mask'].transpose(2, 0, 1).astype('float32')}

        else:
            return {'image': torch.from_numpy(img).transpose(2, 0, 1).astype('float32'),
                    'mask': torch.from_numpy(mask).transpose(2, 0, 1).astype('float32')}
