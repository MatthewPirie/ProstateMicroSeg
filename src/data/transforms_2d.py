# src/data/transforms_2d.py

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


def zscore_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(arr.mean())
    s = float(arr.std())
    if s < eps:
        return arr - m
    return (arr - m) / s


def crop_or_pad_2d(
    arr: np.ndarray,
    target_hw: Tuple[int, int],
    rng: np.random.Generator,
    pad_value: float = 0.0,
    center_yx: Optional[Tuple[int, int]] = None,
    center_jitter: int = 0,
) -> np.ndarray:
    """
    nnU-Net-like sampling:
      - If arr bigger than target: random crop.
        If center_yx provided, crop is chosen so center_yx lands near patch center (with jitter).
      - If arr smaller than target: pad, but placement is random (not symmetric).
        If center_yx provided, pad offset is chosen so center_yx lands near patch center (with jitter).

    arr: (H, W)
    """
    assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
    th, tw = target_hw
    h, w = arr.shape

    if center_yx is not None:
        cy, cx = center_yx
        if center_jitter > 0:
            jy = int(rng.integers(-center_jitter, center_jitter + 1))
            jx = int(rng.integers(-center_jitter, center_jitter + 1))
        else:
            jy = jx = 0
        out_cy = th // 2 + jy
        out_cx = tw // 2 + jx
    else:
        cy = cx = out_cy = out_cx = None

    # ----- Crop in H -----
    if h > th:
        if center_yx is None:
            top = int(rng.integers(0, h - th + 1))
        else:
            # Choose top so that cy maps close to out_cy; clamp to valid crop range
            top = int(np.clip(cy - out_cy, 0, h - th))
        arr = arr[top : top + th, :]
        h = th

    # ----- Crop in W -----
    if w > tw:
        if center_yx is None:
            left = int(rng.integers(0, w - tw + 1))
        else:
            left = int(np.clip(cx - out_cx, 0, w - tw))
        arr = arr[:, left : left + tw]
        w = tw

    # ----- Pad in H -----
    pad_h = th - h
    if pad_h > 0:
        if center_yx is None:
            pad_top = int(rng.integers(0, pad_h + 1))
        else:
            # Choose pad_top so that cy maps close to out_cy; clamp to [0, pad_h]
            pad_top = int(np.clip(out_cy - cy, 0, pad_h))
        pad_bottom = pad_h - pad_top
    else:
        pad_top = pad_bottom = 0

    # ----- Pad in W -----
    pad_w = tw - w
    if pad_w > 0:
        if center_yx is None:
            pad_left = int(rng.integers(0, pad_w + 1))
        else:
            pad_left = int(np.clip(out_cx - cx, 0, pad_w))
        pad_right = pad_w - pad_left
    else:
        pad_left = pad_right = 0

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )

    return arr


def pick_random_foreground_center(
    lbl2d: np.ndarray,
    rng: np.random.Generator,
    thresh: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """
    Returns a random (y, x) coordinate where lbl2d > thresh, or None if no FG.
    """
    fg_yx = np.argwhere(lbl2d > thresh)
    if fg_yx.shape[0] == 0:
        return None
    y, x = fg_yx[int(rng.integers(0, fg_yx.shape[0]))]
    return (int(y), int(x))
