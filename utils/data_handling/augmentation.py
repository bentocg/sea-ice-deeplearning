__all__ = ['get_training_augmentation', 'get_validation_augmentation']

import albumentations as A
from albumentations.pytorch import ToTensor


# center crop used for prediction / validation
def pre_transforms(input_size=256):
    return [A.PadIfNeeded(min_height=input_size, min_width=input_size, always_apply=True, border_mode=0),
            A.CenterCrop(input_size, input_size, p=1)]


# training transformations that don't affect image size
def hard_transforms():
    result = [
        # miscelaneous
        A.IAAAdditiveGaussianNoise(p=0.2),
        #A.CoarseDropout(max_holes=10, max_height=50, max_width=50, min_height=15, min_width=15, p=0.25),

        # brightness
        #A.OneOf(
        #    [
        #        A.RandomBrightnessContrast(p=1),
        #        A.RandomGamma(p=1),
        #    ],
        #    p=0.9,
        #),

        # sharpening / blurring
        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        #
    ]

    return result


# training transformations affecting image size
def resize_transforms(input_size=256):
    result = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=1),
        A.OneOf([A.ShiftScaleRotate(scale_limit=0, rotate_limit=(-45, 45), shift_limit=(-0.1, 0.1),
                                    interpolation=0, border_mode=2, p=0.5),
                 A.ElasticTransform(alpha_affine=20, sigma=30, border_mode=2, p=0.5)]),
        A.PadIfNeeded(min_height=input_size, min_width=input_size, always_apply=True, border_mode=2),
        A.RandomCrop(input_size, input_size, always_apply=True),
    ])

    return result


# helper function to combine groups of augmentations into a single pipeline
def compose(transforms_to_compose):
    result = A.Compose([
        item for sublist in transforms_to_compose for item in sublist
    ])
    return result


# helpers for training and validation augmentation
def get_training_augmentation(input_size=256):
    return compose([resize_transforms(input_size), hard_transforms()])


def get_validation_augmentation(input_size=256):
    return compose([pre_transforms(input_size)])

