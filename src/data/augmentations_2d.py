# src/data/augmentations_2d.py

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    Rand2DElasticd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAdjustContrastd,
)


def build_train_transforms_2d(
    target_hw: Tuple[int, int],
    enabled_augs: Sequence[str] | None = None,
    seed: Optional[int] = None,
    flip_prob: float = 0.5,
    translate_prob: float = 0.15,
    rotate_scale_prob: float = 0.15,
    noise_prob: float = 0.10,
    smooth_prob: float = 0.20,
    shift_prob: float = 0.15,
    scale_prob: float = 0.15,
    contrast_prob: float = 0.20,
    translate_range: Tuple[float, float] = (20.0, 20.0),
    rotate_range: float = 0.350,
    scale_range: Tuple[float, float] = (0.2, 0.2),
) -> Compose:
    """
    Returns a MONAI Compose that expects a dict:
      {"image": torch.Tensor [C,H,W], "label": torch.Tensor [C,H,W]}

    Spatial transforms apply to BOTH image and label (shared random params).
    Intensity transforms apply to image only.

    enabled_augs controls which augmentations are active. Recognised names:
      flip             - left-right and up-down flips (p=0.5 each)
      translate        - random translation in H/W (p=0.15)
      rotate_scale     - random rotation +-20 deg + scale [0.8, 1.2] (p=0.15)
      elastic          - 2D elastic deformation (p=0.10)
      noise            - Gaussian noise (p=0.10)
      smooth           - Gaussian smoothing (p=0.20)
      shift_intensity  - additive intensity shift (p=0.15)
      scale_intensity  - multiplicative intensity scale (p=0.15)
      contrast         - gamma contrast adjustment (p=0.20)

    If enabled_augs is None or empty, returns a no-op Compose.
    """
    enabled = set(enabled_augs or [])
    transforms = []

    # Spatial
    if "flip" in enabled:
        transforms.append(RandFlipd(keys=("image", "label"), prob=flip_prob, spatial_axis=1))  # left-right
        transforms.append(RandFlipd(keys=("image", "label"), prob=flip_prob, spatial_axis=0))  # up-down

    if "translate" in enabled:
        transforms.append(
            RandAffined(
                keys=("image", "label"),
                prob=translate_prob,
                translate_range=(translate_range[0], translate_range[1]),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=target_hw,
            )
        )

    if "rotate_scale" in enabled:
        transforms.append(
            RandAffined(
                keys=("image", "label"),
                prob=rotate_scale_prob,
                rotate_range=(rotate_range,),
                scale_range=(scale_range[0], scale_range[1]),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
                spatial_size=target_hw,
            )
        )

    if "elastic" in enabled:
        transforms.append(
            Rand2DElasticd(
                keys=("image", "label"),
                spacing=(32, 32),
                magnitude_range=(1, 3),
                prob=0.10,
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    # Intensity (image only)
    if "noise" in enabled:
        transforms.append(RandGaussianNoised(keys="image", prob=noise_prob, mean=0.0, std=0.03))

    if "smooth" in enabled:
        transforms.append(
            RandGaussianSmoothd(
                keys="image",
                prob=smooth_prob,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
            )
        )

    if "shift_intensity" in enabled:
        transforms.append(RandShiftIntensityd(keys="image", prob=shift_prob, offsets=(-0.05, 0.05)))

    if "scale_intensity" in enabled:
        transforms.append(RandScaleIntensityd(keys="image", prob=scale_prob, factors=(0.75, 1.25)))

    if "contrast" in enabled:
        transforms.append(RandAdjustContrastd(keys="image", prob=contrast_prob, gamma=(0.7, 1.5)))

    t = Compose(transforms)

    if seed is not None:
        t.set_random_state(seed=seed)

    return t
