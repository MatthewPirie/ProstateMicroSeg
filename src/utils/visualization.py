# src/utils/visualization_3d.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _dice_2d(pred01: np.ndarray, gt01: np.ndarray, eps: float = 1e-8) -> float:
    """
    Dice for binary 2D masks (0/1).
    """
    pred01 = (pred01 > 0).astype(np.uint8)
    gt01 = (gt01 > 0).astype(np.uint8)

    inter = float((pred01 & gt01).sum())
    denom = float(pred01.sum() + gt01.sum())
    if denom == 0.0:
        return 1.0  # both empty -> perfect
    return (2.0 * inter + eps) / (denom + eps)


def _overlay_yellow(base01: np.ndarray, mask01: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    base01: 2D float image in [0,1]
    mask01: 2D binary mask (0/1)
    Returns RGB image with yellow overlay on masked pixels.
    """
    base01 = np.clip(base01, 0.0, 1.0)
    mask = (mask01 > 0).astype(np.float32)

    rgb = np.stack([base01, base01, base01], axis=-1)  # grayscale -> RGB

    # Yellow = (1,1,0)
    yellow = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    rgb = rgb * (1.0 - alpha * mask[..., None]) + yellow[None, None, :] * (alpha * mask[..., None])
    return np.clip(rgb, 0.0, 1.0)


@torch.no_grad()
def infer_full_volume_logits(
    model: torch.nn.Module,
    img: torch.Tensor,  # [1,1,Z,Y,X]
    roi_size: Tuple[int, int, int],
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    amp: bool = True,
) -> torch.Tensor:
    """
    Full-volume inference with stitching (MONAI sliding_window_inference).
    Returns logits with same shape as img: [1,1,Z,Y,X].
    """
    from monai.inferers import sliding_window_inference

    if img.ndim != 5:
        raise RuntimeError(f"Expected img [1,1,Z,Y,X], got {tuple(img.shape)}")

    mode = "gaussian" if gaussian else "constant"
    use_amp = bool(amp) and torch.cuda.is_available() and (img.is_cuda)

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
) -> float:
    """
    Runs full-volume inference, picks one axial slice, and saves a 1x5 grid PNG:
      [image, label, pred, image+label(yellow), image+pred(yellow)]
    Returns the Dice of the selected slice.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    logits = infer_full_volume_logits(
        model=model,
        img=img,
        roi_size=roi_size,
        overlap=overlap,
        sw_batch_size=sw_batch_size,
        gaussian=gaussian,
        amp=amp,
    )

    # Move to CPU numpy for plotting/metrics
    img_np = img.detach().float().cpu().numpy()[0, 0]      # (Z,Y,X)
    lbl_np = lbl.detach().float().cpu().numpy()[0, 0]      # (Z,Y,X)
    prob_np = torch.sigmoid(logits).detach().float().cpu().numpy()[0, 0]  # (Z,Y,X)
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

    # Prepare 2D slices
    base = img_np[z]
    base01 = (base - base.min()) / (base.max() - base.min() + 1e-8)

    gt = (lbl_np[z] > 0.5).astype(np.uint8)
    pr = pred_np[z].astype(np.uint8)

    dice = _dice_2d(pr, gt)

    gt_overlay = _overlay_yellow(base01, gt, alpha=0.35)
    pr_overlay = _overlay_yellow(base01, pr, alpha=0.35)

    # Plot grid
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