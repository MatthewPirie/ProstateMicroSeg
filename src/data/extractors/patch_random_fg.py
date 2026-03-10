# src/data/extractors/patch_random_fg.py

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.data.transforms_3d import center_crop_or_pad_3d  # deterministic behavior stays the same
from src.data.transforms_3d_v2 import (
    pick_random_fg_voxel_3d,
    compute_crop_pad_params_3d_v2,
    apply_crop_pad_3d,
)


def extract(
    img_np: np.ndarray,
    lbl_np: np.ndarray,
    *,
    target_zyx: Tuple[int, int, int],
    rng: np.random.Generator,
    deterministic: bool,
    force_fg: bool,
    fg_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extract a fixed-size 3D patch.

    Behavior:
      - deterministic=True  : center crop/pad (validation-style). Ignores force_fg.
      - deterministic=False : random crop/pad, but if force_fg=True:
                              pick a random FG voxel and place it at a RANDOM
                              location inside the output patch (nnU-Net-ish).

    Returns:
      (img_patch, lbl_patch, meta)
    """
    if img_np.ndim != 3 or lbl_np.ndim != 3:
        raise ValueError(f"Expected 3D arrays (Z,Y,X). Got img {img_np.shape}, lbl {lbl_np.shape}.")

    # Deterministic: always the same centered patch
    if deterministic:
        img_patch = center_crop_or_pad_3d(img_np, target_zyx, pad_value=0.0)
        lbl_patch = center_crop_or_pad_3d(lbl_np, target_zyx, pad_value=0.0)
        meta = {
            "extractor": "patch_random_fg",
            "deterministic": True,
            "force_fg_requested": bool(force_fg),
            "force_fg_used": False,
            "center_zyx": None,
        }
        return img_patch, lbl_patch, meta

    # Training/random mode
    center_zyx: Optional[Tuple[int, int, int]] = None
    force_fg_used = False

    if force_fg:
        center_zyx = pick_random_fg_voxel_3d(lbl_np, rng=rng, thresh=fg_threshold)
        if center_zyx is not None:
            force_fg_used = True

    params = compute_crop_pad_params_3d_v2(
        in_zyx=img_np.shape,
        target_zyx=target_zyx,
        rng=rng,
        center_zyx=center_zyx,  # if None => fully random crop/pad
        out_jitter_zyx=(0, 0, 0),  # keep explicit; can be wired later if you decide to use it
    )

    img_patch = apply_crop_pad_3d(img_np, params, pad_value=0.0)
    lbl_patch = apply_crop_pad_3d(lbl_np, params, pad_value=0.0)

    meta = {
        "extractor": "patch_random_fg",
        "deterministic": False,
        "force_fg_requested": bool(force_fg),
        "force_fg_used": bool(force_fg_used),
        "center_zyx": center_zyx,
        "params": params,
    }
    return img_patch, lbl_patch, meta