# src/train/trainer_3d.py

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    epoch: int,
    steps_per_epoch: int = 250,      # nnU-Net-style fixed optimizer updates per epoch
    log_every: int = 50,
    pin_memory: bool = False,
    scheduler=None,                 # if provided, step per-iteration
    scaler=None, 
    amp: bool = True,
) -> Dict[str, float]:
    """
    One training epoch with a fixed number of optimizer steps.

    Key behavior:
      - We iterate steps_per_epoch times, regardless of dataset length.
      - If the loader iterator runs out (StopIteration), we re-create it and keep going.
    """
    model.train()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0

    running_total = 0.0
    running_steps = 0

    it = iter(train_loader)

    for step in range(1, steps_per_epoch + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        use_amp = bool(amp) and (scaler is not None) and torch.cuda.is_available()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(imgs)
            total_loss, parts = criterion(logits, lbls)

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_val = float(total_loss.item())
        bce_val = float(parts["bce"].detach().item())
        dice_loss_val = float(parts["dice_loss"].detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(
                f"[Epoch {epoch}] step {step}/{steps_per_epoch} | train_total_loss={avg_total:.4f}",
                flush=True,
            )
            running_total = 0.0
            running_steps = 0

    return {
        "train_total_loss": total_sum / max(steps_per_epoch, 1),
        "train_bce": bce_sum / max(steps_per_epoch, 1),
        "train_dice_loss": dice_loss_sum / max(steps_per_epoch, 1),
    }


@torch.no_grad()
def validate_patches_3d(
    model: torch.nn.Module,
    val_loader,              # patch-based val loader
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    num_val_steps: int = 50,  # nnU-Net-style: fixed number of validation iterations
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Patch-level (online) validation.

    Behavior:
      - Runs exactly num_val_steps iterations.
      - Restarts the DataLoader iterator when exhausted.
      - Computes loss and dice on patches (NOT full-volume dice).
    """
    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0

    it = iter(val_loader)

    for step in range(1, num_val_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(val_loader)
            batch = next(it)

        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        logits = model(imgs)
        total_loss, parts = criterion(logits, lbls)

        dice_soft = dice_soft_fn(logits, lbls)
        dice_hard = dice_hard_fn(logits, lbls)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())

    out = {
        "valp_total_loss": total_sum / max(num_val_steps, 1),
        "valp_bce": bce_sum / max(num_val_steps, 1),
        "valp_dice_loss": dice_loss_sum / max(num_val_steps, 1),
        "valp_dice_soft": dice_soft_sum / max(num_val_steps, 1),
        "valp_dice_thr05": dice_hard_sum / max(num_val_steps, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(patches) total_loss={out['valp_total_loss']:.4f} | "
        f"dice_thr05={out['valp_dice_thr05']:.4f}",
        flush=True,
    )
    return out


@torch.no_grad()
def validate_cases_slidingwindow_3d(
    model: torch.nn.Module,
    val_loader,              # expects full volumes, batch_size=1
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    roi_size: Tuple[int, int, int] = (14, 256, 448),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Case-level 3D validation using MONAI sliding-window inference.

    This function evaluates the model on full validation volumes. Each case is
    processed with sliding-window inference so that models trained on local
    3D patches can generate predictions for the entire volume.

    Workflow:
      1) Load a full case from `val_loader` (batch_size=1)
      2) Run MONAI `sliding_window_inference` with window size `roi_size`
      3) Stitch window predictions to form a full-volume prediction
      4) Compute loss and Dice metrics against the full ground-truth label

    Expected input from `val_loader`:
      image: [1, 1, Z, Y, X]
      label: [1, 1, Z, Y, X]

    Returns averaged validation metrics across all cases.
    """
    from monai.inferers import sliding_window_inference

    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_cases = 0

    mode = "gaussian" if gaussian else "constant"

    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=pin_memory)
        lbl = batch["label"].to(device, non_blocking=pin_memory)

        if img.ndim != 5:
            raise RuntimeError(f"Expected img [B,1,Z,Y,X], got {tuple(img.shape)}")
        if img.shape[0] != 1 or img.shape[1] != 1:
            raise RuntimeError(f"Expected batch_size=1 and C=1, got {tuple(img.shape)}")

        logits = sliding_window_inference(
            inputs=img,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )

        total_loss, parts = criterion(logits, lbl)
        dice_soft = dice_soft_fn(logits, lbl)
        dice_hard = dice_hard_fn(logits, lbl)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())
        n_cases += 1

    out = {
        "val_total_loss": total_sum / max(n_cases, 1),
        "val_bce": bce_sum / max(n_cases, 1),
        "val_dice_loss": dice_loss_sum / max(n_cases, 1),
        "val_dice_soft": dice_soft_sum / max(n_cases, 1),
        "val_dice_thr05": dice_hard_sum / max(n_cases, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(cases) total_loss={out['val_total_loss']:.4f} | "
        f"dice_thr05={out['val_dice_thr05']:.4f}",
        flush=True,
    )
    return out

import torch
import torch.nn.functional as F
from typing import Dict, Tuple

def _resize_xy_only_5d(
    x: torch.Tensor,
    target_yx: Tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    """
    Resize only the in-plane dimensions of a 5D tensor [B, C, Z, Y, X].

    Z is preserved exactly. Each slice is resized independently in 2D.
    """
    if x.ndim != 5:
        raise ValueError(f"Expected [B, C, Z, Y, X], got shape {tuple(x.shape)}")

    target_y, target_x = int(target_yx[0]), int(target_yx[1])
    b, c, z, y, xdim = x.shape

    if y == target_y and xdim == target_x:
        return x

    # [B, C, Z, Y, X] -> [B*Z, C, Y, X]
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

    # [B*Z, C, target_y, target_x] -> [B, C, Z, target_y, target_x]
    out = out2d.reshape(b, z, c, target_y, target_x).permute(0, 2, 1, 3, 4).contiguous()
    return out

@torch.no_grad()
def validate_cases_fullinplane_3d(
    model: torch.nn.Module,
    val_loader,              # expects full volumes, batch_size=1
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    roi_size: Tuple[int, int, int] = (14, 256, 256),
    overlap: float = 0.5,
    sw_batch_size: int = 1,
    gaussian: bool = True,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Case-level 3D validation for full in-plane extractors.

    This function is intended for models trained on contiguous Z windows with
    full in-plane content resized to a fixed XY size, for example
    `full_inplane_zwindow`.

    Workflow:
      1) Load a full case [1, 1, Z, Y, X]
      2) Resize only XY to match the model input space, keep Z unchanged
      3) Run MONAI sliding-window inference with roi_size=(z_window, target_y, target_x)
      4) Compute loss and Dice metrics in the resized space

    Expected input from `val_loader`:
      image: [1, 1, Z, Y, X]
      label: [1, 1, Z, Y, X]

    Returns averaged validation metrics across all cases.
    """
    from monai.inferers import sliding_window_inference

    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_cases = 0

    z_window, target_y, target_x = int(roi_size[0]), int(roi_size[1]), int(roi_size[2])
    mode = "gaussian" if gaussian else "constant"

    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]
        lbl = batch["label"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]

        if img.ndim != 5:
            raise RuntimeError(f"Expected img [B,1,Z,Y,X], got {tuple(img.shape)}")
        if lbl.ndim != 5:
            raise RuntimeError(f"Expected lbl [B,1,Z,Y,X], got {tuple(lbl.shape)}")
        if img.shape[0] != 1 or img.shape[1] != 1:
            raise RuntimeError(f"Expected batch_size=1 and C=1 for image, got {tuple(img.shape)}")
        if lbl.shape[0] != 1 or lbl.shape[1] != 1:
            raise RuntimeError(f"Expected batch_size=1 and C=1 for label, got {tuple(lbl.shape)}")

        # Resize only XY, preserve Z exactly.
        img_rs = _resize_xy_only_5d(
            img.float(),
            (target_y, target_x),
            mode="bilinear",
        )
        lbl_rs = _resize_xy_only_5d(
            lbl.float(),
            (target_y, target_x),
            mode="nearest",
        )

        # Safety binarization after nearest-neighbor resize
        lbl_rs = (lbl_rs > 0.5).float()

        if img_rs.shape[2] < z_window:
            raise RuntimeError(
                f"Case has Z={img_rs.shape[2]} which is smaller than roi_size[0]={z_window}. "
                f"Either reduce patch_z or add Z-padding logic for case validation."
            )

        logits = sliding_window_inference(
            inputs=img_rs,
            roi_size=(z_window, target_y, target_x),
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=float(overlap),
            mode=mode,
        )

        total_loss, parts = criterion(logits, lbl_rs)
        dice_soft = dice_soft_fn(logits, lbl_rs)
        dice_hard = dice_hard_fn(logits, lbl_rs)

        total_sum += float(total_loss.item())
        bce_sum += float(parts["bce"].detach().item())
        dice_loss_sum += float(parts["dice_loss"].detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())
        n_cases += 1

    out = {
        "val_total_loss": total_sum / max(n_cases, 1),
        "val_bce": bce_sum / max(n_cases, 1),
        "val_dice_loss": dice_loss_sum / max(n_cases, 1),
        "val_dice_soft": dice_soft_sum / max(n_cases, 1),
        "val_dice_thr05": dice_hard_sum / max(n_cases, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(cases/full-inplane) total_loss={out['val_total_loss']:.4f} | "
        f"dice_thr05={out['val_dice_thr05']:.4f}",
        flush=True,
    )
    return out

