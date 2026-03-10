# src/data/augmentations_3d.py

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    RandAffined,
    RandAdjustContrastd,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
)

# - Samples are dicts with keys: "image" and "label"
# - Tensors are channel-first: [C, Z, Y, X]
# - Stored arrays are (Z, Y, X) with anisotropic Z
# - Keep spatial transforms conservative


def _deg2rad(x: float) -> float:
    return float(x) * (np.pi / 180.0)


def build_train_transforms_3d(
    *,
    patch_zyx: Tuple[int, int, int],
    enabled_augs: Sequence[str] | None = None,
    mirror_axes: Sequence[int] = (0, 1, 2),

    # rotation + scale affine
    p_rotscale: float = 0.05,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    rot_deg_about_z: float = 10.0,
    rot_deg_about_y: float = 0.0,
    rot_deg_about_x: float = 0.0,

    # translation affine
    p_translate: float = 0.05,
    translate_range_zyx: Tuple[float, float, float] = (0.0, 16.0, 16.0),

    # other augs
    p_noise: float = 0.05,
    p_brightness: float = 0.10,
    p_contrast: float = 0.10,
    noise_std: float = 0.05,
    brightness_factor: float = 0.10,
    contrast_gamma: Tuple[float, float] = (0.9, 1.1),
) -> Compose:
    enabled = set(enabled_augs or [])
    transforms = []

    # -------------------------
    # Rotation + scale
    # -------------------------
    if "rotscale" in enabled:
        rotate_range = (
            _deg2rad(rot_deg_about_z),
            _deg2rad(rot_deg_about_y),
            _deg2rad(rot_deg_about_x),
        )

        lo, hi = float(scale_range[0]), float(scale_range[1])
        scale_range_add = (lo - 1.0, hi - 1.0)

        transforms.append(
            RandAffined(
                keys=("image", "label"),
                prob=float(p_rotscale),
                spatial_size=tuple(int(v) for v in patch_zyx),
                rotate_range=rotate_range,
                scale_range=(scale_range_add, scale_range_add, scale_range_add),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    # -------------------------
    # Translation only
    # -------------------------
    if "translate" in enabled:
        transforms.append(
            RandAffined(
                keys=("image", "label"),
                prob=float(p_translate),
                spatial_size=tuple(int(v) for v in patch_zyx),
                translate_range=tuple(float(v) for v in translate_range_zyx),
                rotate_range=(0.0, 0.0, 0.0),
                scale_range=(0.0, 0.0, 0.0),
                mode=("bilinear", "nearest"),
                padding_mode="zeros",
            )
        )

    # -------------------------
    # Flip
    # -------------------------
    if "flip" in enabled:
        for ax in mirror_axes:
            transforms.append(
                RandFlipd(
                    keys=("image", "label"),
                    prob=0.5,
                    spatial_axis=int(ax),
                )
            )

    # -------------------------
    # Noise
    # -------------------------
    if "noise" in enabled:
        transforms.append(
            RandGaussianNoised(
                keys="image",
                prob=float(p_noise),
                mean=0.0,
                std=float(noise_std),
            )
        )

    # -------------------------
    # Brightness
    # -------------------------
    if "brightness" in enabled:
        transforms.append(
            RandScaleIntensityd(
                keys="image",
                prob=float(p_brightness),
                factors=float(brightness_factor),
            )
        )

    # -------------------------
    # Contrast
    # -------------------------
    if "contrast" in enabled:
        transforms.append(
            RandAdjustContrastd(
                keys="image",
                prob=float(p_contrast),
                gamma=tuple(float(v) for v in contrast_gamma),
            )
        )

    return Compose(transforms)


def build_val_transforms_3d() -> Compose:
    return Compose([])