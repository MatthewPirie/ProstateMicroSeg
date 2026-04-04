# src/data/extractors/full_inplane_zstrided.py

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np


def _pad_z_to_len(vol: np.ndarray, target_z: int, pad_value: float) -> np.ndarray:
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
    has_fg = (lbl_np > fg_threshold).reshape(lbl_np.shape[0], -1).any(axis=1)
    return np.nonzero(has_fg)[0]


def _evenly_spaced_indices(Z: int, Z_out: int) -> np.ndarray:
    """
    Return length-Z_out integer indices spanning [0, Z-1] as evenly as possible.
    For Z_out <= Z, indices are unique and increasing.
    """
    if Z_out <= 1:
        return np.array([0], dtype=np.int64)
    # Use linspace then round to nearest int; with endpoint included this spans the full depth.
    idx = np.linspace(0, Z - 1, num=Z_out)
    idx = np.round(idx).astype(np.int64)
    # Ensure in bounds and monotonic non-decreasing
    idx = np.clip(idx, 0, Z - 1)
    return idx

from typing import Dict, Tuple
import numpy as np


def extract(
    img_np: np.ndarray,
    lbl_np: np.ndarray,
    *,
    target_zyx: Tuple[int, int, int],
    rng: np.random.Generator,     # unused, kept for consistent interface
    deterministic: bool,          # unused (this extractor is deterministic by design)
    force_fg: bool,              # unused
    fg_threshold: float,          # unused
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Full in-plane extraction with strided (evenly spaced) Z selection.

    Output:
      img_out, lbl_out: (Z_out, Y, X) where Y,X are unchanged (full in-plane).

    Rules:
      - Always select Z_out slices spanning the full depth [0 .. Z-1] (no interpolation).
      - Ignores deterministic/force_fg because strided selection is already deterministic.
      - If Z < Z_out: pad along Z first to reach Z_out, then select indices.
    """
    if img_np.ndim != 3 or lbl_np.ndim != 3:
        raise ValueError(f"Expected 3D arrays (Z,Y,X). Got img {img_np.shape}, lbl {lbl_np.shape}.")
    if img_np.shape != lbl_np.shape:
        raise ValueError(f"img/lbl shape mismatch: img {img_np.shape}, lbl {lbl_np.shape}.")

    Z_out = int(target_zyx[0])
    if Z_out <= 0:
        raise ValueError(f"Z_out must be > 0, got {Z_out}.")

    Z, Y, X = img_np.shape

    padded = False
    if Z < Z_out:
        img_np = _pad_z_to_len(img_np, Z_out, pad_value=0.0)
        lbl_np = _pad_z_to_len(lbl_np, Z_out, pad_value=0.0)
        padded = True
        Z = img_np.shape[0]  # now equals Z_out

    idx = _evenly_spaced_indices(Z, Z_out)

    img_out = img_np[idx, :, :]
    lbl_out = lbl_np[idx, :, :]

    meta = {
        "extractor": "full_inplane_zstrided",
        "z_out": int(Z_out),
        "idx": idx.astype(int).tolist(),
        "padded_z": bool(padded),
        "orig_shape": (int(Z), int(Y), int(X)),
        "ignored_args": {
            "deterministic": bool(deterministic),
            "force_fg": bool(force_fg),
            "fg_threshold": float(fg_threshold),
        },
    }
    return img_out, lbl_out, meta