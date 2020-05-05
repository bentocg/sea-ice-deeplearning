import logging
from glob import glob
from os import listdir
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir,  outline_dir=None, augmentation=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        if outline_dir:
            self.outline_dir = outline_dir
        else:
            self.outline_dir = None

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.augmentation = augmentation
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = img_nd.reshape((img_nd.shape[0], img_nd.shape[1], 1)).astype(np.uint8)

        return img_nd

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.tif')
        img_file = glob(self.imgs_dir + idx + '.tif')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask =self.preprocess(Image.open(mask_file[0]))
        img = self.preprocess(Image.open(img_file[0]))

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        if self.outline_dir:
            outline_file = glob(self.outline_dir + idx + '.tif')
            assert len(outline_file) == 1, \
                f'Either no outline or multiple outlines found for the ID {idx}: {outline_file}'
            outline = Image.open(outline_file[0])

        else:
            outline = np.zeros(np.array(mask).shape)

        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            return {'image': sample['image'].transpose((2, 0, 1)), 'mask': sample['mask'].transpose((2, 0, 1))}

        else:
            return {'image': torch.from_numpy(img).float(), 'mask': torch.from_numpy(mask).float()}
