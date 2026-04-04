# ProstateMicroSeg/src/data/transforms_2d.py

from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np


def zscore_normalize_2d(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    [DETERMINISTIC] Z-score normalize a single 2D slice using its own mean and std.

    Produces identical output for identical input — no randomness.

    Used in: training and validation, applied per-slice before any spatial
    crop/pad step.
    """
    m = float(arr.mean())
    s = float(arr.std())
    if s < eps:
        return arr - m
    return (arr - m) / s


def sample_foreground_center_2d(
    lbl2d: np.ndarray,
    rng: np.random.Generator,
    thresh: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """
    [STOCHASTIC] Sample a random foreground pixel coordinate (y, x) from a 2D label mask.

    Returns None when the slice contains no foreground. Output varies with rng state.

    Used in: training only, per-slice. Provides a foreground anchor for
    compute_stochastic_crop_pad_plan_2d when force-foreground sampling is active
    (nnU-Net-style). Never called during validation or inference.
    """
    fg_yx = np.argwhere(lbl2d > thresh)
    if fg_yx.shape[0] == 0:
        return None
    y, x = fg_yx[int(rng.integers(0, fg_yx.shape[0]))]
    return (int(y), int(x))


def compute_stochastic_crop_pad_plan_2d(
    in_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    rng: np.random.Generator,
    center_yx: Optional[Tuple[int, int]] = None,
    center_jitter: int = 0,
) -> Dict[str, int]:
    """
    [STOCHASTIC] Compute a shared crop+pad plan mapping (H, W) to (target_h, target_w).

    Produces one plan dict that must be applied to both image and label via
    apply_crop_pad_plan_2d to preserve spatial alignment. Output varies with rng state.

    center_yx=None  : fully random crop start and random pad placement.
    center_yx given : places that pixel near the patch center (±center_jitter pixels),
                      implementing nnU-Net-style forced-foreground centering.

    Returns dict keys: top, left, crop_h, crop_w,
                       pad_top, pad_bottom, pad_left, pad_right,
                       target_h, target_w.

    Used in: training only, per-slice. Call once per (image, label) pair; pass the
    same plan to apply_crop_pad_plan_2d for both arrays. Never called during
    validation or inference (use deterministic_center_crop_pad_2d instead).
    """
    th, tw = target_hw
    h, w = in_hw

    if center_yx is not None:
        cy, cx = center_yx
        jy = int(rng.integers(-center_jitter, center_jitter + 1)) if center_jitter > 0 else 0
        jx = int(rng.integers(-center_jitter, center_jitter + 1)) if center_jitter > 0 else 0
        out_cy = th // 2 + jy
        out_cx = tw // 2 + jx
    else:
        cy = cx = out_cy = out_cx = None

    crop_h = min(h, th)
    crop_w = min(w, tw)

    if h > th:
        if center_yx is None:
            top = int(rng.integers(0, h - th + 1))
        else:
            top = int(np.clip(cy - out_cy, 0, h - th))
    else:
        top = 0

    if w > tw:
        if center_yx is None:
            left = int(rng.integers(0, w - tw + 1))
        else:
            left = int(np.clip(cx - out_cx, 0, w - tw))
    else:
        left = 0

    pad_h = th - crop_h
    pad_w = tw - crop_w

    if pad_h > 0:
        if center_yx is None:
            pad_top = int(rng.integers(0, pad_h + 1))
        else:
            cy_in_crop = cy - top
            pad_top = int(np.clip(out_cy - cy_in_crop, 0, pad_h))
        pad_bottom = pad_h - pad_top
    else:
        pad_top = pad_bottom = 0

    if pad_w > 0:
        if center_yx is None:
            pad_left = int(rng.integers(0, pad_w + 1))
        else:
            cx_in_crop = cx - left
            pad_left = int(np.clip(out_cx - cx_in_crop, 0, pad_w))
        pad_right = pad_w - pad_left
    else:
        pad_left = pad_right = 0

    return {
        "top": top,
        "left": left,
        "crop_h": crop_h,
        "crop_w": crop_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "target_h": th,
        "target_w": tw,
    }


def apply_crop_pad_plan_2d(
    arr: np.ndarray, p: Dict[str, int], pad_value: float = 0.0
) -> np.ndarray:
    """
    [DETERMINISTIC] Apply a precomputed crop+pad plan to a 2D array.

    Given the same plan dict, always produces the same output. Call with the
    identical plan for both image and label to maintain spatial alignment.

    Used in: training only, per-slice. Applied after compute_stochastic_crop_pad_plan_2d
    to both the image and label arrays. Never called during validation or inference.
    """
    top = p["top"]
    left = p["left"]
    crop_h = p["crop_h"]
    crop_w = p["crop_w"]

    out = arr[top : top + crop_h, left : left + crop_w]

    if p["pad_top"] or p["pad_bottom"] or p["pad_left"] or p["pad_right"]:
        out = np.pad(
            out,
            pad_width=((p["pad_top"], p["pad_bottom"]), (p["pad_left"], p["pad_right"])),
            mode="constant",
            constant_values=pad_value,
        )

    # Safety: enforce exact size
    out = out[: p["target_h"], : p["target_w"]]
    assert out.shape == (p["target_h"], p["target_w"]), f"{out.shape} != {(p['target_h'], p['target_w'])}"
    return out


def deterministic_center_crop_pad_2d(
    arr: np.ndarray, target_hw: Tuple[int, int], pad_value: float = 0.0
) -> np.ndarray:
    """
    [DETERMINISTIC] Center crop and/or pad a single 2D slice to an exact target size.

    Produces identical output for identical input — no randomness.
      - If the slice is larger than target on an axis: take a centered crop.
      - If the slice is smaller than target on an axis: apply symmetric zero-padding.

    Used in: validation and inference preprocessing, per-slice. Replaces the
    stochastic training pipeline (compute_stochastic_crop_pad_plan_2d +
    apply_crop_pad_plan_2d) for reproducible, unbiased evaluation. For whole-volume
    torch-based preprocessing use deterministic_center_crop_pad_3d_torch from
    transforms_3d.py instead.
    """
    assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
    th, tw = target_hw
    h, w = arr.shape

    # Center crop
    if h > th:
        top = (h - th) // 2
        arr = arr[top : top + th, :]
    if w > tw:
        left = (w - tw) // 2
        arr = arr[:, left : left + tw]

    # Center pad
    h2, w2 = arr.shape
    pad_h = max(th - h2, 0)
    pad_w = max(tw - w2, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    return arr
