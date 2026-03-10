# src/data/extractors/full_inplane_zwindow.py

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


def _pad_z_to_window(
    vol: np.ndarray,
    target_z: int,
    pad_value: float,
) -> np.ndarray:
    """
    Pad a (Z,Y,X) volume along Z to reach target_z, symmetric padding.
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
    Return array of z indices where the label has any foreground.
    """
    # (Z,Y,X) -> (Z,) boolean: any FG in each slice
    has_fg = (lbl_np > fg_threshold).reshape(lbl_np.shape[0], -1).any(axis=1)
    return np.nonzero(has_fg)[0]


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
    Full in-plane extraction with a Z window.

    Output:
      img_out, lbl_out: (Z_window, Y, X) where Y,X are unchanged (full in-plane).

    Rules:
      - deterministic=True:
          take centered Z window (or pad if Z < Z_window)
      - deterministic=False and force_fg=False:
          choose z0 uniformly from [0, Z - Z_window] (or pad if needed)
      - deterministic=False and force_fg=True:
          pick a random FG slice z_fg, then choose z0 uniformly from the valid
          start positions that ensure z_fg is inside [z0, z0+Z_window-1].
          If no FG exists, fall back to uniform random z0.
    """
    if img_np.ndim != 3 or lbl_np.ndim != 3:
        raise ValueError(f"Expected 3D arrays (Z,Y,X). Got img {img_np.shape}, lbl {lbl_np.shape}.")
    if img_np.shape != lbl_np.shape:
        raise ValueError(f"img/lbl shape mismatch: img {img_np.shape}, lbl {lbl_np.shape}.")

    z_win = int(target_zyx[0])
    if z_win <= 0:
        raise ValueError(f"Z window must be > 0, got {z_win}.")

    Z, Y, X = img_np.shape

    # If the case is shorter than the window, pad in Z first (symmetric).
    if Z < z_win:
        img_pad = _pad_z_to_window(img_np, z_win, pad_value=0.0)
        lbl_pad = _pad_z_to_window(lbl_np, z_win, pad_value=0.0)
        meta = {
            "extractor": "full_inplane_zwindow",
            "deterministic": bool(deterministic),
            "force_fg_requested": bool(force_fg),
            "force_fg_used": False,
            "z0": 0,
            "z_window": z_win,
            "padded_z": True,
            "orig_shape": (int(Z), int(Y), int(X)),
            "out_shape": tuple(int(v) for v in img_pad.shape),
        }
        return img_pad[:z_win, :, :], lbl_pad[:z_win, :, :], meta

    # Now Z >= z_win, so we can choose a valid z0 in [0, Z - z_win]
    max_start = Z - z_win

    if deterministic:
        z0 = max_start // 2
        meta = {
            "extractor": "full_inplane_zwindow",
            "deterministic": True,
            "force_fg_requested": bool(force_fg),
            "force_fg_used": False,
            "z0": int(z0),
            "z_window": int(z_win),
            "padded_z": False,
            "orig_shape": (int(Z), int(Y), int(X)),
        }
        return img_np[z0 : z0 + z_win, :, :], lbl_np[z0 : z0 + z_win, :, :], meta

    # Training/random mode
    force_fg_used = False
    z0: Optional[int] = None

    if force_fg:
        fg_idx = _fg_slices(lbl_np, fg_threshold=fg_threshold)
        if fg_idx.size > 0:
            z_fg = int(fg_idx[int(rng.integers(0, fg_idx.size))])

            # Valid z0 range that includes z_fg:
            # z0 <= z_fg <= z0 + z_win - 1  =>
            # z0 in [z_fg-(z_win-1), z_fg]
            low = max(0, z_fg - (z_win - 1))
            high = min(z_fg, max_start)

            # If low/high are valid, sample uniformly in that range.
            if low <= high:
                z0 = int(rng.integers(low, high + 1))
                force_fg_used = True

    # Fallback to uniform random window
    if z0 is None:
        z0 = int(rng.integers(0, max_start + 1))

    meta = {
        "extractor": "full_inplane_zwindow",
        "deterministic": False,
        "force_fg_requested": bool(force_fg),
        "force_fg_used": bool(force_fg_used),
        "z0": int(z0),
        "z_window": int(z_win),
        "padded_z": False,
        "orig_shape": (int(Z), int(Y), int(X)),
    }

    return img_np[z0 : z0 + z_win, :, :], lbl_np[z0 : z0 + z_win, :, :], meta