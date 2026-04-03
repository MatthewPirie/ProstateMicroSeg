# src/utils/visualization_3d.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _dice_2d(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice for binary 2D masks (0/1).
    """
    pred01 = (pred01 > 0).astype(np.uint8)
    gt01 = (gt01 > 0).astype(np.uint8)

    inter = float((pred01 & gt01).sum())
    denom = float(pred01.sum() + gt01.sum())
    if denom == 0.0:
        return 1.0
    return (2.0 * inter + eps) / (denom + eps)


def _overlay_yellow(base01: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    base01: 2D float image in [0,1]
    mask01: 2D binary mask (0/1)
    Returns RGB image with yellow overlay on masked pixels.
    """
    base01 = np.clip(base01, 0.0, 1.0)
    mask = (mask01 > 0).astype(np.float32)

    rgb = np.stack([base01, base01, base01], axis=-1)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    rgb = rgb * (1.0 - alpha * mask[..., None]) + yellow[None, None, :] * (alpha * mask[..., None])
    return np.clip(rgb, 0.0, 1.0)


def _resize_xy_only_5d(
    x: torch.Tensor,
    target_yx: Tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    """
    Resize only XY of a 5D tensor [B, C, Z, Y, X].
    Z is preserved exactly.
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B, C, Z, Y, X], got shape {tuple(x.shape)}")

    target_y, target_x = int(target_yx[0]), int(target_yx[1])
    b, c, z, y, xdim = x.shape

    if y == target_y and xdim == target_x:
        return x

    x2d = x.permute(0, 2, 1, 3, 4).reshape(b * z, c, y, xdim)

    if mode == "nearest":
        out2d = F.interpolate(
            x2d,
            size=(target_y, target_x),
            mode="nearest",
        )
    elif mode in {"bilinear", "bicubic"}:
        out2d = F.interpolate(
            x2d,
            size=(target_y, target_x),
            mode=mode,
            align_corners=False,
        )
    else:
        raise ValueError(f"Unsupported resize mode: {mode}")

    out = out2d.reshape(b, z, c, target_y, target_x).permute(0, 2, 1, 3, 4).contiguous()
    return out


@torch.no_grad()
def _infer_full_volume_logits_slidingwindow(
    model: torch.nn.Module,
    img: torch.Tensor,  # [1,1,Z,Y,X]
    roi_size: Tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    amp: bool = True,
) -> torch.Tensor:
    """
    Full-volume inference on the raw full case using MONAI sliding-window inference.
    """
    from monai.inferers import sliding_window_inference

    if img.ndim != 5:
        raise RuntimeError(f"Expected img [1,1,Z,Y,X], got {tuple(img.shape)}")

    mode = "gaussian" if gaussian else "constant"
    use_amp = bool(amp) and torch.cuda.is_available() and img.is_cuda

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )
    return logits


@torch.no_grad()
def _infer_full_volume_logits_fullinplane(
    model: torch.nn.Module,
    img: torch.Tensor,  # [1,1,Z,Y,X]
    roi_size: Tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full-volume inference for full-inplane models.

    Resizes XY to roi_size[1:] first, keeps Z unchanged, then runs
    sliding-window inference in that resized space.

    Returns:
      logits_rs: [1,1,Z,target_y,target_x]
      img_rs:    [1,1,Z,target_y,target_x]
    """
    from monai.inferers import sliding_window_inference

    if img.ndim != 5:
        raise RuntimeError(f"Expected img [1,1,Z,Y,X], got {tuple(img.shape)}")

    z_window, target_y, target_x = int(roi_size[0]), int(roi_size[1]), int(roi_size[2])

    img_rs = _resize_xy_only_5d(
        img.float(),
        (target_y, target_x),
        mode="bilinear",
    )

    if img_rs.shape[2] < z_window:
        raise RuntimeError(
            f"Case has Z={img_rs.shape[2]} which is smaller than roi_size[0]={z_window}."
        )

    mode = "gaussian" if gaussian else "constant"
    use_amp = bool(amp) and torch.cuda.is_available() and img_rs.is_cuda

    with torch.cuda.amp.autocast(enabled=use_amp):
        logits_rs = sliding_window_inference(
            inputs=img_rs,
            roi_size=(z_window, target_y, target_x),
            sw_batch_size=int(sw_batch_size),
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )

    return logits_rs, img_rs


@torch.no_grad()
def save_val_case_slice_grid_png(
    *,
    model: torch.nn.Module,
    img: torch.Tensor,   # [1,1,Z,Y,X], already on device
    lbl: torch.Tensor,   # [1,1,Z,Y,X], already on device
    out_png: str | Path,
    roi_size: Tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    thr: float = 0.5,
    slice_index: Optional[int] = None,
    choose: str = "middle",  # "middle" | "best_dice"
    amp: bool = True,
    vis_mode: str = "slidingwindow",  # "slidingwindow" | "fullinplane"
) -> float:
    """
    Runs case-level inference, picks one axial slice, and saves a 1x5 PNG grid:
      [image, label, pred, image+label(yellow), image+pred(yellow)]

    vis_mode:
      - "slidingwindow": standard raw full-volume MONAI sliding-window inference
      - "fullinplane": resize XY first, then run sliding-window inference in resized space

    Returns the Dice of the selected slice.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if vis_mode == "slidingwindow":
        logits = _infer_full_volume_logits_slidingwindow(
            model=model,
            img=img,
            roi_size=roi_size,
            overlap=overlap,
            sw_batch_size=sw_batch_size,
            gaussian=gaussian,
            amp=amp,
        )
        img_vis = img
        lbl_vis = lbl

    elif vis_mode == "fullinplane":
        print("DEBUGL: vis_mode is fullinplane")
        logits, img_vis = _infer_full_volume_logits_fullinplane(
            model=model,
            img=img,
            roi_size=roi_size,
            overlap=overlap,
            sw_batch_size=sw_batch_size,
            gaussian=gaussian,
            amp=amp,
        )
        lbl_vis = _resize_xy_only_5d(
            lbl.float(),
            (int(roi_size[1]), int(roi_size[2])),
            mode="nearest",
        )
        lbl_vis = (lbl_vis > 0.5).float()

    else:
        raise ValueError(f"Unsupported vis_mode: {vis_mode}")

    img_np = img_vis.detach().float().cpu().numpy()[0, 0]   # (Z,Y,X)
    lbl_np = lbl_vis.detach().float().cpu().numpy()[0, 0]   # (Z,Y,X)
    prob_np = torch.sigmoid(logits).detach().float().cpu().numpy()[0, 0]
    pred_np = (prob_np >= float(thr)).astype(np.uint8)

    zdim = img_np.shape[0]
    if slice_index is not None:
        z = int(slice_index)
        z = max(0, min(z, zdim - 1))
    else:
        if choose == "best_dice":
            best = (-1.0, 0)
            for zi in range(zdim):
                d = _dice_2d(pred_np[zi], (lbl_np[zi] > 0.5).astype(np.uint8))
                if d > best[0]:
                    best = (d, zi)
            z = int(best[1])
        else:
            z = zdim // 2

    base = img_np[z]
    base01 = (base - base.min()) / (base.max() - base.min() + 1e-8)

    gt = (lbl_np[z] > 0.5).astype(np.uint8)
    pr = pred_np[z].astype(np.uint8)

    dice = _dice_2d(pr, gt)

    gt_overlay = _overlay_yellow(base01, gt, alpha=0.35)
    pr_overlay = _overlay_yellow(base01, pr, alpha=0.35)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(base01, cmap="gray")
    axes[0].set_title(f"Image (z={z})")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("Label")
    axes[1].axis("off")

    axes[2].imshow(pr, cmap="gray")
    axes[2].set_title(f"Pred (thr={thr:.2f})")
    axes[2].axis("off")

    axes[3].imshow(gt_overlay)
    axes[3].set_title("Label overlay (yellow)")
    axes[3].axis("off")

    axes[4].imshow(pr_overlay)
    axes[4].set_title(f"Pred overlay (Dice={dice:.3f})")
    axes[4].axis("off")

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return float(dice)