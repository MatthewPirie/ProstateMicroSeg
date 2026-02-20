# ProstateMicroSeg/src/train/trainer_2d.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from src.data.transforms_2d import center_crop_or_pad_2d


def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    epoch: int,
    log_every: int = 100,
    pin_memory: bool = False,
    max_steps: int = 250,   # nnU-Net-style: fixed optimizer steps per epoch
    scheduler=None,         # if provided, step per iteration
) -> Dict[str, float]:
    """
    One training epoch with a fixed number of optimizer steps (nnU-Net-style).

    Returns:
        {
          "train_total_loss": float,
          "train_bce": float,
          "train_dice_loss": float
        }
    """
    model.train()

    # For console logging (windowed average)
    running_total = 0.0
    running_steps = 0

    # Epoch averages
    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    n_steps = 0

    for step, batch in enumerate(train_loader, start=1):
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)

        logits = model(imgs)
        total_loss, parts = criterion(logits, lbls)

        # Backprop + update
        total_loss.backward()
        optimizer.step()

        # IMPORTANT: step LR scheduler per optimizer step (iteration-based scheduling)
        # if scheduler is not None:
        #     scheduler.step()

        # Convert to python floats for aggregation
        total_val = float(total_loss.item())
        bce_val = float(parts["bce"].detach().item())
        dice_loss_val = float(parts["dice_loss"].detach().item())

        total_sum += total_val
        bce_sum += bce_val
        dice_loss_sum += dice_loss_val
        n_steps += 1

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg_total = running_total / max(running_steps, 1)
            print(
                f"[Epoch {epoch}] step {step}/{max_steps} | "
                f"train_total_loss={avg_total:.4f}"
            )
            running_total = 0.0
            running_steps = 0

        # Stop epoch after fixed number of optimizer updates
        if max_steps is not None and step >= max_steps:
            break

    return {
        "train_total_loss": total_sum / max(n_steps, 1),
        "train_bce": bce_sum / max(n_steps, 1),
        "train_dice_loss": dice_loss_sum / max(n_steps, 1),
    }

@torch.no_grad()
def validate_case_level_2d(
    model: torch.nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    target_hw: Tuple[int, int] = (896, 1408),
    slice_batch_size: int = 16,
    pin_memory: bool = False,
) -> Dict[str, float]:
    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_cases = 0

    for batch in val_loader:
        # IMPORTANT: keep on CPU first (batch tensors from DataLoader are CPU by default)
        img3d = batch["image"]  # [1,1,Z,Y,X] on CPU
        lbl3d = batch["label"]  # [1,1,Z,Y,X] on CPU

        # [1,1,Z,Y,X] -> [Z,1,Y,X]
        img2d_all = img3d.squeeze(0).permute(1, 0, 2, 3).contiguous()
        lbl2d_all = lbl3d.squeeze(0).permute(1, 0, 2, 3).contiguous()

        # Deterministic center crop/pad in TORCH (fast, no numpy, no per-slice loop)
        img2d_all = center_crop_or_pad_2d_torch(img2d_all, target_hw, pad_value=0.0)
        lbl2d_all = center_crop_or_pad_2d_torch(lbl2d_all, target_hw, pad_value=0.0)

        # Move once
        img2d_all = img2d_all.float().to(device, non_blocking=pin_memory)
        lbl2d_all = lbl2d_all.float().to(device, non_blocking=pin_memory)

        Z = img2d_all.shape[0]

        # forward in chunks
        logits_chunks = []
        for start in range(0, Z, slice_batch_size):
            end = min(start + slice_batch_size, Z)
            logits_chunks.append(model(img2d_all[start:end]))
        logits2d_all = torch.cat(logits_chunks, dim=0)  # [Z,1,H,W]

        # stack back to volume
        logits3d = logits2d_all.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # [1,1,Z,H,W]
        lbl3d_eval = lbl2d_all.permute(1, 0, 2, 3).unsqueeze(0).contiguous()   # [1,1,Z,H,W]

        total_loss, parts = criterion(logits3d, lbl3d_eval)
        dice_soft = dice_soft_fn(logits3d, lbl3d_eval)
        dice_hard = dice_hard_fn(logits3d, lbl3d_eval)

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

    print(f"[Epoch {epoch}] VAL(case,2D) total_loss={out['val_total_loss']:.4f} | dice_thr05={out['val_dice_thr05']:.4f}")
    return out


def center_crop_or_pad_2d_torch(x: torch.Tensor, target_hw: Tuple[int, int], pad_value: float = 0.0) -> torch.Tensor:
    """
    x: [Z, 1, H, W] on CPU or GPU
    returns: [Z, 1, target_h, target_w]
    """
    th, tw = target_hw
    Z, C, H, W = x.shape
    assert C == 1

    # center crop if too large
    if H > th:
        top = (H - th) // 2
        x = x[:, :, top:top + th, :]
        H = th
    if W > tw:
        left = (W - tw) // 2
        x = x[:, :, :, left:left + tw]
        W = tw

    # center pad if too small
    pad_h = max(th - H, 0)
    pad_w = max(tw - W, 0)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)

    return x


def save_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    """
    Minimal checkpoint saver.
    Stores auxiliary info under payload["extra"] to avoid key collisions.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra if extra is not None else {},
    }

    torch.save(payload, ckpt_path)

@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader,
    criterion,  # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    dice_soft_fn,  # expects: dice_soft_fn(logits, targets) -> scalar tensor
    dice_hard_fn,  # expects: dice_hard_fn(logits, targets) -> scalar tensor
    epoch: int,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Validation loop.

    Returns:
        {
          "val_total_loss": float,
          "val_bce": float,
          "val_dice_loss": float,
          "val_dice_soft": float,
          "val_dice_thr05": float
        }
    """
    model.eval()

    total_sum = 0.0
    bce_sum = 0.0
    dice_loss_sum = 0.0
    dice_soft_sum = 0.0
    dice_hard_sum = 0.0
    n_batches = 0

    for batch in val_loader:
        imgs = batch["image"].to(device, non_blocking=pin_memory)
        lbls = batch["label"].to(device, non_blocking=pin_memory)

        logits = model(imgs)

        total_loss, parts = criterion(logits, lbls)
        bce = parts["bce"]
        dice_loss = parts["dice_loss"]

        dice_soft = dice_soft_fn(logits, lbls)
        dice_hard = dice_hard_fn(logits, lbls)

        total_sum += float(total_loss.item())
        bce_sum += float(bce.detach().item())
        dice_loss_sum += float(dice_loss.detach().item())
        dice_soft_sum += float(dice_soft.detach().item())
        dice_hard_sum += float(dice_hard.detach().item())

        n_batches += 1

    val_total = total_sum / max(n_batches, 1)
    val_bce = bce_sum / max(n_batches, 1)
    val_dice_loss = dice_loss_sum / max(n_batches, 1)
    val_dice_soft = dice_soft_sum / max(n_batches, 1)
    val_dice_thr05 = dice_hard_sum / max(n_batches, 1)

    print(
        f"[Epoch {epoch}] "
        f"VAL total_loss={val_total:.4f} | "
        f"VAL dice_thr05={val_dice_thr05:.4f}"
    )

    return {
        "val_total_loss": val_total,
        "val_bce": val_bce,
        "val_dice_loss": val_dice_loss,
        "val_dice_soft": val_dice_soft,
        "val_dice_thr05": val_dice_thr05,
    }

