# src/data/transforms_2d.py

from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np

def zscore_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(arr.mean())
    s = float(arr.std())
    if s < eps:
        return arr - m
    return (arr - m) / s

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

def compute_crop_pad_params_2d(
    in_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
    rng: np.random.Generator,
    center_yx: Optional[Tuple[int, int]] = None,
    center_jitter: int = 0,
) -> Dict[str, int]:
    """
    Compute ONE shared crop+pad plan to map an input (H,W) to exactly target (th,tw).

    Returns:
      top, left, crop_h, crop_w, pad_top, pad_bottom, pad_left, pad_right
    """
    th, tw = target_hw
    h, w = in_hw

    # Choose desired location of the 'center' inside the output patch (with jitter)
    if center_yx is not None:
        cy, cx = center_yx
        jy = int(rng.integers(-center_jitter, center_jitter + 1)) if center_jitter > 0 else 0
        jx = int(rng.integers(-center_jitter, center_jitter + 1)) if center_jitter > 0 else 0
        out_cy = th // 2 + jy
        out_cx = tw // 2 + jx
    else:
        cy = cx = out_cy = out_cx = None

    # Decide how much of the input we will take BEFORE padding
    crop_h = min(h, th)
    crop_w = min(w, tw)

    # Choose crop start so that center maps near desired output center (or random)
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

    # Now compute padding needed AFTER cropping to reach target
    pad_h = th - crop_h
    pad_w = tw - crop_w

    if pad_h > 0:
        if center_yx is None:
            pad_top = int(rng.integers(0, pad_h + 1))
        else:
            # where should the input center land in the cropped region? It's at (cy - top)
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


def apply_crop_pad_2d(arr: np.ndarray, p: Dict[str, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Apply a precomputed crop+pad plan. This keeps image and label perfectly aligned.
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

# Used for validation
def center_crop_or_pad_2d(arr: np.ndarray, target_hw: Tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Deterministic center crop/pad (for validation).
    - If larger than target: take centered crop.
    - If smaller than target: symmetric (center) padding.
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


# Redundant function (can be deleted)
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