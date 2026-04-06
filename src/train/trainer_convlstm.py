# src/train/trainer_convlstm.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from src.train.trainer_3d import _resize_xy_only_5d


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch_convlstm(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion,                        # expects: total_loss, parts = criterion(logits, targets)
    device: torch.device,
    epoch: int,
    steps_per_epoch: int = 250,       # nnU-Net-style: fixed optimizer updates per epoch
    log_every: int = 50,
    pin_memory: bool = False,
    scheduler=None,                   # if provided, stepped per iteration
    scaler=None,
    amp: bool = True,
    grad_clip_norm: float = 1.0,      # max gradient norm; set <= 0.0 to disable
) -> Dict[str, float]:
    """
    One training epoch for SegmentationConvLSTM with a fixed number of optimizer steps.

    Identical to the 3D trainer loop except gradient clipping is applied before
    each optimizer step. Hidden state is implicitly zero-initialised per batch,
    matching the random-window training regime.

    Returns:
        {
          "train_total_loss": float,
          "train_bce":        float,
          "train_dice_loss":  float,
        }
    """
    model.train()

    total_sum = 0.0
    bce_sum   = 0.0
    dice_sum  = 0.0

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
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_val = float(total_loss.item())
        bce_val   = float(parts["bce"].detach().item())
        dice_val  = float(parts["dice_loss"].detach().item())

        total_sum += total_val
        bce_sum   += bce_val
        dice_sum  += dice_val

        running_total += total_val
        running_steps += 1

        if step == 1 or (log_every > 0 and step % log_every == 0):
            avg = running_total / max(running_steps, 1)
            print(
                f"[Epoch {epoch}] step {step}/{steps_per_epoch} | train_total_loss={avg:.4f}",
                flush=True,
            )
            running_total = 0.0
            running_steps = 0

    return {
        "train_total_loss": total_sum / max(steps_per_epoch, 1),
        "train_bce":        bce_sum   / max(steps_per_epoch, 1),
        "train_dice_loss":  dice_sum  / max(steps_per_epoch, 1),
    }


# ---------------------------------------------------------------------------
# Patch-level validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_patches_convlstm(
    model: torch.nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    num_val_steps: int = 50,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Patch-level (online) validation — runs exactly num_val_steps iterations,
    restarting the DataLoader iterator when exhausted. Computes loss and Dice
    on patches, not full volumes.
    """
    model.eval()

    total_sum = 0.0
    bce_sum   = 0.0
    dice_sum  = 0.0
    soft_sum  = 0.0
    hard_sum  = 0.0

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
        bce_sum   += float(parts["bce"].detach().item())
        dice_sum  += float(parts["dice_loss"].detach().item())
        soft_sum  += float(dice_soft.detach().item())
        hard_sum  += float(dice_hard.detach().item())

    out = {
        "valp_total_loss": total_sum / max(num_val_steps, 1),
        "valp_bce":        bce_sum   / max(num_val_steps, 1),
        "valp_dice_loss":  dice_sum  / max(num_val_steps, 1),
        "valp_dice_soft":  soft_sum  / max(num_val_steps, 1),
        "valp_dice_thr05": hard_sum  / max(num_val_steps, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(patches) total_loss={out['valp_total_loss']:.4f} | "
        f"dice_thr05={out['valp_dice_thr05']:.4f}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Case-level validation helpers
# ---------------------------------------------------------------------------

def _forward_window_with_state(
    model,
    window: torch.Tensor,
    hidden_state,
):
    """
    Forward one Z window through SegmentationConvLSTM, threading hidden state.

    Args:
        model:        SegmentationConvLSTM instance
        window:       (B, C, Z_window, H, W)
        hidden_state: List[(h, c)] from previous window, or None for cold start

    Returns:
        logits:       (B, out_channels, Z_window, H, W)
        hidden_state: List[(h, c)] — final state of this window, to pass to next
    """
    skips, bottleneck = model._encode_frames(window)
    temporal_outputs, last_state_list = model.temporal(bottleneck, hidden_state)
    temporal_feats = temporal_outputs[0]
    logits = model._decode_frames(temporal_feats, skips)
    return logits, last_state_list


# ---------------------------------------------------------------------------
# Case-level validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_cases_convlstm(
    model: torch.nn.Module,
    val_loader,                             # full volumes, batch_size=1
    criterion,
    device: torch.device,
    dice_soft_fn,
    dice_hard_fn,
    epoch: int,
    roi_size: Tuple[int, int, int] = (16, 256, 256),  # (z_window, target_y, target_x)
    overlap: float = 0.5,                   # used only when use_hidden_carryover=False
    sw_batch_size: int = 1,                 # used only when use_hidden_carryover=False
    gaussian: bool = True,                  # used only when use_hidden_carryover=False
    use_hidden_carryover: bool = False,
    pin_memory: bool = False,
) -> Dict[str, float]:
    """
    Case-level 3D validation for SegmentationConvLSTM trained with full_inplane_zwindow.

    Workflow:
      1) Load full case [1, 1, Z, Y, X]
      2) Resize XY to (target_y, target_x), preserve Z
      3a) use_hidden_carryover=False (default):
            MONAI sliding_window_inference in Z — matches training conditions exactly.
      3b) use_hidden_carryover=True:
            Non-overlapping Z windows processed in order; LSTM hidden state from
            each window is passed into the next. Gives later slices access to
            accumulated temporal context from earlier in the volume.

    Args:
        roi_size:             (z_window, target_y, target_x)
        overlap:              Sliding-window overlap fraction (carryover=False only).
        use_hidden_carryover: Thread LSTM state across Z windows during inference.

    Returns averaged validation metrics across all cases.
    """
    from monai.inferers import sliding_window_inference

    model.eval()

    z_window, target_y, target_x = int(roi_size[0]), int(roi_size[1]), int(roi_size[2])
    mode = "gaussian" if gaussian else "constant"

    total_sum = 0.0
    bce_sum   = 0.0
    dice_sum  = 0.0
    soft_sum  = 0.0
    hard_sum  = 0.0
    n_cases   = 0

    for batch in val_loader:
        img = batch["image"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]
        lbl = batch["label"].to(device, non_blocking=pin_memory)  # [1,1,Z,Y,X]

        if img.ndim != 5 or img.shape[0] != 1 or img.shape[1] != 1:
            raise RuntimeError(f"Expected img [1,1,Z,Y,X], got {tuple(img.shape)}")
        if lbl.ndim != 5 or lbl.shape[0] != 1 or lbl.shape[1] != 1:
            raise RuntimeError(f"Expected lbl [1,1,Z,Y,X], got {tuple(lbl.shape)}")

        img_rs = _resize_xy_only_5d(img.float(), (target_y, target_x), mode="bilinear")
        lbl_rs = _resize_xy_only_5d(lbl.float(), (target_y, target_x), mode="nearest")
        lbl_rs = (lbl_rs > 0.5).float()

        z_total = img_rs.shape[2]

        if z_total < z_window:
            # Volume is shorter than the training window.  Run the model on
            # the full volume directly — the ConvLSTM handles any sequence
            # length and no padding / replication is introduced.
            logits = model(img_rs)

        elif use_hidden_carryover:
            # ---------------------------------------------------------------
            # Non-overlapping windows in Z order; thread LSTM hidden state.
            # ---------------------------------------------------------------
            hidden_state = None
            logits_chunks = []

            for z_start in range(0, z_total, z_window):
                z_end = min(z_start + z_window, z_total)
                window = img_rs[:, :, z_start:z_end, :, :]
                logits_w, hidden_state = _forward_window_with_state(
                    model, window, hidden_state
                )
                logits_chunks.append(logits_w)

            logits = torch.cat(logits_chunks, dim=2)  # [1, 1, Z, H, W]

        else:
            # ---------------------------------------------------------------
            # MONAI sliding-window inference — matches training conditions.
            # ---------------------------------------------------------------
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
        bce_sum   += float(parts["bce"].detach().item())
        dice_sum  += float(parts["dice_loss"].detach().item())
        soft_sum  += float(dice_soft.detach().item())
        hard_sum  += float(dice_hard.detach().item())
        n_cases   += 1

    carryover_tag = "/carryover" if use_hidden_carryover else ""
    out = {
        "val_total_loss": total_sum / max(n_cases, 1),
        "val_bce":        bce_sum   / max(n_cases, 1),
        "val_dice_loss":  dice_sum  / max(n_cases, 1),
        "val_dice_soft":  soft_sum  / max(n_cases, 1),
        "val_dice_thr05": hard_sum  / max(n_cases, 1),
    }

    print(
        f"[Epoch {epoch}] VAL(cases/convlstm{carryover_tag}) "
        f"total_loss={out['val_total_loss']:.4f} | "
        f"dice_thr05={out['val_dice_thr05']:.4f}",
        flush=True,
    )
    return out

