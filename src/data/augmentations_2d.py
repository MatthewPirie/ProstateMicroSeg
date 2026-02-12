# src/data/augmentations_2d.py

from __future__ import annotations

from typing import Optional, Tuple

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    Rand2DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,   # gamma-like augmentation
)

def get_train_transforms_2d(
    target_hw: Tuple[int, int],
    seed: Optional[int] = None,
):
    """
    Returns a MONAI Compose that expects a dict:
      {"image": torch.Tensor [C,H,W], "label": torch.Tensor [C,H,W]}

    Spatial transforms apply to BOTH image and label (shared random params).
    Intensity transforms apply to image only.
    Patch size is preserved via spatial_size=target_hw in RandAffined.
    """

    t = Compose(
        [
            # Spatial 
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),  # left-right
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),  # up-down

            RandAffined(
                keys=("image", "label"),
                prob=0.15,
                rotate_range=(0.350,),          # +/- 20 degrees
                scale_range=(0.2, 0.2),        # approx [0.8, 1.2]
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=target_hw,
            ),

            Rand2DElasticd(
                keys=("image", "label"),
                spacing=(32, 32),
                magnitude_range=(1, 3),
                prob=0.10,                     # low to keep it cheap
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            ),

            #  Intensity (image only) 
            RandGaussianNoised(keys="image", prob=0.10, mean=0.0, std=0.03),

            RandGaussianSmoothd(
                keys="image",
                prob=0.20,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            ),

            RandShiftIntensityd(keys="image", prob=0.15, offsets=(-0.05, 0.05)),

            RandScaleIntensityd(keys="image", prob=0.15, factors=(0.75, 1.25)),

            # gamma-like contrast transform
            RandAdjustContrastd(keys="image", prob=0.20, gamma=(0.7, 1.5)),
        ]
    )

    if seed is not None:
        t.set_random_state(seed=seed)

    return t
