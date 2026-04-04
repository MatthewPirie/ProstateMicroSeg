# src/data/transforms_3d.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def sample_foreground_voxel_3d(
    lbl: np.ndarray,
    rng: np.random.Generator,
    thresh: float = 0.5,
) -> Optional[Tuple[int, int, int]]:
    """
    [STOCHASTIC] Sample a random foreground voxel coordinate (z, y, x) from a 3D label volume.

    Returns None when the volume contains no foreground. Output varies with rng state.

    Used in: training only, whole-volume. Provides a foreground anchor for
    compute_stochastic_crop_pad_plan_3d when force-foreground sampling is active
    (nnU-Net-style). Never called during validation or inference.
    """
    fg = np.argwhere(lbl > thresh)
    if fg.shape[0] == 0:
        return None
    cz, cy, cx = fg[int(rng.integers(0, fg.shape[0]))]
    return int(cz), int(cy), int(cx)


def compute_stochastic_crop_pad_plan_3d(
    in_zyx: Tuple[int, int, int],
    target_zyx: Tuple[int, int, int],
    rng: np.random.Generator,
    center_zyx: Optional[Tuple[int, int, int]] = None,
    out_jitter_zyx: Tuple[int, int, int] = (0, 0, 0),
) -> Dict[str, int]:
    """
    [STOCHASTIC] Compute a shared crop+pad plan mapping (Z, Y, X) to (tz, ty, tx).

    Produces one plan dict that must be applied to both image and label via
    apply_crop_pad_plan_3d to preserve spatial alignment. Output varies with rng state.

    center_zyx=None  : fully random crop start and random pad placement on every axis.
    center_zyx given : the specified foreground voxel is placed at a uniformly random
                       position inside the output patch (nnU-Net-style v2 behaviour —
                       random-in-patch rather than patch-center placement).
                       out_jitter_zyx adds an optional extra perturbation (pass (0,0,0)
                       to disable).

    Returns dict keys: z0, y0, x0 (crop starts),
                       crop_z, crop_y, crop_x (crop extents),
                       pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1 (padding),
                       target_z, target_y, target_x (guaranteed output shape).

    Used in: training only, whole-volume. Call once per (image, label) volume pair;
    pass the same plan to apply_crop_pad_plan_3d for both arrays. Never called
    during validation or inference (use deterministic_center_crop_or_pad_3d or
    deterministic_center_crop_pad_3d_torch instead).
    """
    (z, y, x) = in_zyx
    (tz, ty, tx) = target_zyx

    if center_zyx is None:
        c_in = (None, None, None)
        c_out = (None, None, None)
    else:
        oz = int(rng.integers(0, tz))
        oy = int(rng.integers(0, ty))
        ox = int(rng.integers(0, tx))

        c_in = center_zyx
        c_out = (oz, oy, ox)

    def _axis_plan(
        n_in: int,
        n_tgt: int,
        c_in_ax: Optional[int],
        c_out_ax: Optional[int],
    ) -> Tuple[int, int, int, int]:
        """
        Returns: start, crop_len, pad_before, pad_after

        - If n_in > n_tgt: crop a window of length n_tgt, choosing 'start' randomly
          or so that the foreground voxel lands at c_out_ax.
        - If n_in <= n_tgt: keep the whole axis (crop_len = n_in, start = 0) and
          choose padding offsets randomly or to align the foreground voxel.
        """
        crop_len = min(n_in, n_tgt)

        if n_in > n_tgt:
            if c_in_ax is None:
                start = int(rng.integers(0, n_in - n_tgt + 1))
            else:
                start = int(np.clip(c_in_ax - c_out_ax, 0, n_in - n_tgt))
        else:
            start = 0

        pad_total = n_tgt - crop_len
        if pad_total > 0:
            if c_in_ax is None:
                pad_before = int(rng.integers(0, pad_total + 1))
            else:
                c_in_crop = c_in_ax - start
                pad_before = int(np.clip(c_out_ax - c_in_crop, 0, pad_total))
            pad_after = pad_total - pad_before
        else:
            pad_before = 0
            pad_after = 0

        return start, crop_len, pad_before, pad_after

    z0, crop_z, pz0, pz1 = _axis_plan(z, tz, c_in[0], c_out[0])
    y0, crop_y, py0, py1 = _axis_plan(y, ty, c_in[1], c_out[1])
    x0, crop_x, px0, px1 = _axis_plan(x, tx, c_in[2], c_out[2])

    return {
        "z0": z0,
        "y0": y0,
        "x0": x0,
        "crop_z": crop_z,
        "crop_y": crop_y,
        "crop_x": crop_x,
        "pad_z0": pz0,
        "pad_z1": pz1,
        "pad_y0": py0,
        "pad_y1": py1,
        "pad_x0": px0,
        "pad_x1": px1,
        "target_z": tz,
        "target_y": ty,
        "target_x": tx,
    }


def apply_crop_pad_plan_3d(
    vol: np.ndarray, p: Dict[str, int], pad_value: float = 0.0
) -> np.ndarray:
    """
    [DETERMINISTIC] Apply a precomputed crop+pad plan to a 3D volume.

    Given the same plan dict, always produces the same output. Call with the
    identical plan for both image and label to maintain spatial alignment.

    Used in: training, whole-volume. Applied after compute_stochastic_crop_pad_plan_3d
    to both the image and label volumes. May also be used in inference pipelines
    that pre-build a fixed plan, though deterministic_center_crop_or_pad_3d is the
    typical deterministic entry point.
    """
    out = vol[
        p["z0"] : p["z0"] + p["crop_z"],
        p["y0"] : p["y0"] + p["crop_y"],
        p["x0"] : p["x0"] + p["crop_x"],
    ]

    if any(
        p[k]
        for k in (
            "pad_z0",
            "pad_z1",
            "pad_y0",
            "pad_y1",
            "pad_x0",
            "pad_x1",
        )
    ):
        out = np.pad(
            out,
            pad_width=((p["pad_z0"], p["pad_z1"]), (p["pad_y0"], p["pad_y1"]), (p["pad_x0"], p["pad_x1"])),
            mode="constant",
            constant_values=pad_value,
        )

    out = out[: p["target_z"], : p["target_y"], : p["target_x"]]
    assert out.shape == (p["target_z"], p["target_y"], p["target_x"]), (
        f"{out.shape} != {(p['target_z'], p['target_y'], p['target_x'])}"
    )
    return out


def deterministic_center_crop_or_pad_3d(
    vol: np.ndarray, target_zyx: Tuple[int, int, int], pad_value: float = 0.0
) -> np.ndarray:
    """
    [DETERMINISTIC] Center crop and/or pad a 3D numpy volume to an exact target size.

    Produces identical output for identical input — no randomness.
      - If larger than target on an axis: take a centered crop on that axis.
      - If smaller than target on an axis: apply symmetric zero-padding on that axis.

    Used in: validation and inference preprocessing, whole-volume (numpy path).
    Replaces the stochastic training pipeline for reproducible evaluation. For
    torch-tensor inputs (e.g. stacked 2D volumes in [Z, 1, H, W] format) use
    deterministic_center_crop_pad_3d_torch instead.
    """
    assert vol.ndim == 3, f"Expected 3D vol, got {vol.ndim}D"
    tz, ty, tx = target_zyx
    z, y, x = vol.shape

    # Center crop
    if z > tz:
        z0 = (z - tz) // 2
        vol = vol[z0 : z0 + tz, :, :]
    if y > ty:
        y0 = (y - ty) // 2
        vol = vol[:, y0 : y0 + ty, :]
    if x > tx:
        x0 = (x - tx) // 2
        vol = vol[:, :, x0 : x0 + tx]

    # Center pad
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


def deterministic_center_crop_pad_3d_torch(
    x: "torch.Tensor",
    target_hw: Tuple[int, int],
    pad_value: float = 0.0,
) -> "torch.Tensor":
    """
    [DETERMINISTIC] Center crop and/or pad a stacked 2D volume (torch tensor) to an
    exact spatial size.

    Operates on an entire volume in one call — all slices are cropped/padded
    identically, preserving inter-slice spatial consistency. Runs on the tensor's
    existing device (CPU or GPU) without any numpy conversion.

    Args:
        x         : Tensor of shape [Z, 1, H, W].
        target_hw : (target_h, target_w) desired output spatial size.
        pad_value : constant fill value used for padding.

    Returns:
        Tensor of shape [Z, 1, target_h, target_w].

    Used in: validation and inference preprocessing, whole-volume (torch path).
    Preferred over calling deterministic_center_crop_pad_2d per-slice when the
    full volume is already available as a [Z, 1, H, W] tensor (avoids per-slice
    python loops and numpy round-trips). Never called during training.
    """
    import torch.nn.functional as F

    th, tw = target_hw
    Z, C, H, W = x.shape
    assert C == 1

    # Center crop if too large
    if H > th:
        top = (H - th) // 2
        x = x[:, :, top:top + th, :]
        H = th
    if W > tw:
        left = (W - tw) // 2
        x = x[:, :, :, left:left + tw]
        W = tw

    # Center pad if too small
    pad_h = max(th - H, 0)
    pad_w = max(tw - W, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    return x