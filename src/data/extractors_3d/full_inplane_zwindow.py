# src/data/extractors/full_inplane_zwindow.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _pad_z_to_window(
    vol: np.ndarray,
    target_z: int,
    pad_value: float,
) -> np.ndarray:
    """
    Pad a (Z, Y, X) volume along Z to reach target_z, using symmetric padding.
    """
    z, y, x = vol.shape
    if z >= target_z:
        return vol

    pad_total = target_z - z
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before

    return np.pad(
        vol,
        pad_width=((pad_before, pad_after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def _fg_slices(lbl_np: np.ndarray, fg_threshold: float) -> np.ndarray:
    """
    Return z indices where the label contains any foreground.
    """
    has_fg = (lbl_np > fg_threshold).reshape(lbl_np.shape[0], -1).any(axis=1)
    return np.nonzero(has_fg)[0]


def _resize_xy_slicewise(
    vol: np.ndarray,
    target_yx: Tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    """
    Resize each slice in a (Z, Y, X) volume to (target_y, target_x),
    without changing Z.

    Uses torch interpolation slice-wise by treating Z as batch:
      input  -> [Z, 1, Y, X]
      output -> [Z, 1, target_y, target_x]
    """
    if vol.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X), got shape {vol.shape}.")

    target_y, target_x = int(target_yx[0]), int(target_yx[1])
    z, y, x = vol.shape

    if y == target_y and x == target_x:
        return vol

    vol_t = torch.from_numpy(vol.copy()).unsqueeze(1).float()                                                                                         

    if mode == "nearest":
        out_t = F.interpolate(
            vol_t,
            size=(target_y, target_x),
            mode="nearest",
        )
    elif mode in {"bilinear", "bicubic"}:
        out_t = F.interpolate(
            vol_t,
            size=(target_y, target_x),
            mode=mode,
            align_corners=False,
        )
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    out = out_t.squeeze(1).cpu().numpy()  # [Z, target_y, target_x]
    return out


def extract(
    img_np: np.ndarray,
    lbl_np: np.ndarray,
    *,
    target_zyx: Tuple[int, int, int],
    rng: np.random.Generator,
    deterministic: bool,
    force_fg: bool,
    fg_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full in-plane extraction with a Z window, followed by XY resizing.

    Logic:
      1) Choose a contiguous Z window of length z_window
      2) Do NOT resize/interpolate in Z
      3) Resize XY of the extracted subvolume to (target_y, target_x)

    Output:
      img_out, lbl_out: (z_window, target_y, target_x)

    Rules:
      - deterministic=True:
          take centered Z window (or pad in Z first if Z < z_window)
      - deterministic=False and force_fg=False:
          choose z0 uniformly from [0, Z - z_window] (or pad in Z first if needed)
      - deterministic=False and force_fg=True:
          pick a random FG slice z_fg, then choose z0 uniformly from the valid
          start positions that ensure z_fg is inside [z0, z0 + z_window - 1].
          If no FG exists, fall back to uniform random z0.
    """
    if img_np.ndim != 3 or lbl_np.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (Z, Y, X). Got img {img_np.shape}, lbl {lbl_np.shape}."
        )
    if img_np.shape != lbl_np.shape:
        raise ValueError(f"img/lbl shape mismatch: img {img_np.shape}, lbl {lbl_np.shape}.")

    z_window = int(target_zyx[0])
    target_y = int(target_zyx[1])
    target_x = int(target_zyx[2])

    if z_window <= 0:
        raise ValueError(f"z_window must be > 0, got {z_window}.")
    if target_y <= 0 or target_x <= 0:
        raise ValueError(f"target_y and target_x must be > 0, got {(target_y, target_x)}.")

    Z, Y, X = img_np.shape

    padded_z = False
    force_fg_used = False

    # ------------------------------------------------------------------
    # Step 1: ensure we can return exactly z_window slices
    # ------------------------------------------------------------------
    if Z < z_window:
        img_work = _pad_z_to_window(img_np, z_window, pad_value=0.0)
        lbl_work = _pad_z_to_window(lbl_np, z_window, pad_value=0.0)
        padded_z = True
        Z_work = z_window
    else:
        img_work = img_np
        lbl_work = lbl_np
        Z_work = Z

    # ------------------------------------------------------------------
    # Step 2: choose contiguous Z window
    # ------------------------------------------------------------------
    max_start = Z_work - z_window

    if deterministic:
        z0 = max_start // 2
    else:
        z0: Optional[int] = None

        if force_fg:
            fg_idx = _fg_slices(lbl_work, fg_threshold=fg_threshold)
            if fg_idx.size > 0:
                z_fg = int(fg_idx[int(rng.integers(0, fg_idx.size))])

                # Need z_fg inside [z0, z0 + z_window - 1]
                low = max(0, z_fg - (z_window - 1))
                high = min(z_fg, max_start)

                if low <= high:
                    z0 = int(rng.integers(low, high + 1))
                    force_fg_used = True

        if z0 is None:
            z0 = int(rng.integers(0, max_start + 1))

    img_crop = img_work[z0 : z0 + z_window, :, :]
    lbl_crop = lbl_work[z0 : z0 + z_window, :, :]

    # ------------------------------------------------------------------
    # Step 3: resize XY only, keep Z unchanged
    # ------------------------------------------------------------------
    img_out = _resize_xy_slicewise(
        img_crop.astype(np.float32, copy=False),
        (target_y, target_x),
        mode="bilinear",
    )

    lbl_out = _resize_xy_slicewise(
        lbl_crop.astype(np.float32, copy=False),
        (target_y, target_x),
        mode="nearest",
    )

    meta = {
        "extractor": "full_inplane_zwindow",
        "deterministic": bool(deterministic),
        "force_fg_requested": bool(force_fg),
        "force_fg_used": bool(force_fg_used),
        "z0": int(z0),
        "z_window": int(z_window),
        "padded_z": bool(padded_z),
        "orig_shape": (int(Z), int(Y), int(X)),
        "pre_resize_shape": tuple(int(v) for v in img_crop.shape),
        "out_shape": tuple(int(v) for v in img_out.shape),
        "target_xy": (int(target_y), int(target_x)),
    }

    return img_out, lbl_out, meta