# src/data/transforms_3d.py

from __future__ import annotations
from typing import Optional, Tuple, Dict
import numpy as np


def zscore_normalize_3d(vol: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(vol.mean())
    s = float(vol.std())
    if s < eps:
        return vol - m
    return (vol - m) / s


def pick_random_foreground_center_3d(
    lbl: np.ndarray,
    rng: np.random.Generator,
    thresh: float = 0.5,
) -> Optional[Tuple[int, int, int]]:
    fg = np.argwhere(lbl > thresh)
    if fg.shape[0] == 0:
        return None
    cz, cy, cx = fg[int(rng.integers(0, fg.shape[0]))]
    return int(cz), int(cy), int(cx)


def compute_crop_pad_params_3d(
    in_zyx: Tuple[int, int, int],
    target_zyx: Tuple[int, int, int],
    rng: np.random.Generator,
    center_zyx: Optional[Tuple[int, int, int]] = None,
    center_jitter_zyx: Tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, int]:
    """
    Sample one shared crop+pad plan for (Z,Y,X) -> (tz,ty,tx).
    Use SAME plan for image and label to keep alignment.
    """
    (z, y, x) = in_zyx
    (tz, ty, tx) = target_zyx

    # Output "desired center" inside the target patch (with jitter)
    if center_zyx is None:
        out_center = (None, None, None)
    else:
        j = []
        for jmax in center_jitter_zyx:
            if jmax > 0:
                j.append(int(rng.integers(-jmax, jmax + 1)))
            else:
                j.append(0)
        out_center = (tz // 2 + j[0], ty // 2 + j[1], tx // 2 + j[2])

    def _axis_plan(n_in: int, n_tgt: int, c_in: Optional[int], c_out: Optional[int]) -> Tuple[int, int, int, int]:
        """
        Returns: start, crop_len, pad_before, pad_after
        """
        crop_len = min(n_in, n_tgt)

        # Crop start
        if n_in > n_tgt:
            if c_in is None:
                start = int(rng.integers(0, n_in - n_tgt + 1))
            else:
                start = int(np.clip(c_in - c_out, 0, n_in - n_tgt))
        else:
            start = 0

        # Pad after crop
        pad_total = n_tgt - crop_len
        if pad_total > 0:
            if c_in is None:
                pad_before = int(rng.integers(0, pad_total + 1))
            else:
                c_in_crop = c_in - start
                pad_before = int(np.clip(c_out - c_in_crop, 0, pad_total))
            pad_after = pad_total - pad_before
        else:
            pad_before = pad_after = 0

        return start, crop_len, pad_before, pad_after

    if center_zyx is None:
        c_in = (None, None, None)
        c_out = (None, None, None)
    else:
        c_in = center_zyx
        c_out = out_center

    z0, crop_z, pz0, pz1 = _axis_plan(z, tz, c_in[0], c_out[0])
    y0, crop_y, py0, py1 = _axis_plan(y, ty, c_in[1], c_out[1])
    x0, crop_x, px0, px1 = _axis_plan(x, tx, c_in[2], c_out[2])

    return {
        "z0": z0, "y0": y0, "x0": x0,
        "crop_z": crop_z, "crop_y": crop_y, "crop_x": crop_x,
        "pad_z0": pz0, "pad_z1": pz1,
        "pad_y0": py0, "pad_y1": py1,
        "pad_x0": px0, "pad_x1": px1,
        "target_z": tz, "target_y": ty, "target_x": tx,
    }


def apply_crop_pad_3d(vol: np.ndarray, p: Dict[str, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Apply a precomputed plan from compute_crop_pad_params_3d.
    """
    out = vol[
        p["z0"] : p["z0"] + p["crop_z"],
        p["y0"] : p["y0"] + p["crop_y"],
        p["x0"] : p["x0"] + p["crop_x"],
    ]

    if any(p[k] for k in ("pad_z0", "pad_z1", "pad_y0", "pad_y1", "pad_x0", "pad_x1")):
        out = np.pad(
            out,
            pad_width=((p["pad_z0"], p["pad_z1"]), (p["pad_y0"], p["pad_y1"]), (p["pad_x0"], p["pad_x1"])),
            mode="constant",
            constant_values=pad_value,
        )

    out = out[: p["target_z"], : p["target_y"], : p["target_x"]]
    assert out.shape == (p["target_z"], p["target_y"], p["target_x"]), f"{out.shape} != {(p['target_z'], p['target_y'], p['target_x'])}"
    return out


def center_crop_or_pad_3d(vol: np.ndarray, target_zyx: Tuple[int, int, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Deterministic center crop/pad for validation.
    """
    assert vol.ndim == 3, f"Expected 3D vol, got {vol.ndim}D"
    tz, ty, tx = target_zyx
    z, y, x = vol.shape

    # Crop
    if z > tz:
        z0 = (z - tz) // 2
        vol = vol[z0 : z0 + tz, :, :]
    if y > ty:
        y0 = (y - ty) // 2
        vol = vol[:, y0 : y0 + ty, :]
    if x > tx:
        x0 = (x - tx) // 2
        vol = vol[:, :, x0 : x0 + tx]

    # Pad
    z2, y2, x2 = vol.shape
    pz = max(tz - z2, 0)
    py = max(ty - y2, 0)
    px = max(tx - x2, 0)

    pz0, pz1 = pz // 2, pz - (pz // 2)
    py0, py1 = py // 2, py - (py // 2)
    px0, px1 = px // 2, px - (px // 2)

    if pz or py or px:
        vol = np.pad(
            vol,
            pad_width=((pz0, pz1), (py0, py1), (px0, px1)),
            mode="constant",
            constant_values=pad_value,
        )

    assert vol.shape == (tz, ty, tx), f"Got {vol.shape}, expected {(tz, ty, tx)}"
    return vol
