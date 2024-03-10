import os
import cv2
from albumentations import (
    Compose,
    HorizontalFlip,
    Affine,
    RandomBrightnessContrast,
    HueSaturationValue,
    Blur,
    ShiftScaleRotate,
    CLAHE,
    GaussNoise,
    CoarseDropout,
    Resize,
)
import numpy as np

augmentations = Compose(
    [
        HorizontalFlip(p=0.5),
        Affine(
            scale=None,
            translate_percent=None,
            translate_px=None,
            rotate=None,
            shear=10,
            p=0.5,
        ),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5
        ),
        Blur(blur_limit=3, p=0.3),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7),
        CLAHE(clip_limit=2, p=0.5),
        GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        CoarseDropout(
            max_holes=8,
            max_height=8,
            max_width=8,
            min_holes=1,
            min_height=1,
            min_width=1,
            p=0.5,
        ),
    ]
)


def augment_and_save_images(
    image_directory, output_directory, augmentations, num_augmentations=3
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(image_directory):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(image_directory, filename)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            for i in range(num_augmentations):
                augmented = augmentations(image=image)
                augmented_image = augmented["image"]

                save_path = os.path.join(output_directory, f"aug_{i}_{filename}")
                augmented_image = cv2.cvtColor(
                    augmented_image, cv2.COLOR_RGB2BGR
                )  # Convert back to BGR for saving
                cv2.imwrite(save_path, augmented_image)


image_directory = "../data/sign_vs_no/sign"
output_directory = "../data/sign_vs_no/train/sign"
augment_and_save_images(
    image_directory, output_directory, augmentations, num_augmentations=4
)
