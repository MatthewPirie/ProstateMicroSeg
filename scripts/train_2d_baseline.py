# ProstateMicroSeg/scripts/train_2d_baseline.py

# Debug: first thing that runs — if this is slow, the delay is in the SLURM/shell step
import sys
print("[train_2d_baseline] Python started, importing...", flush=True)

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow: from src...

import argparse
print("[train_2d_baseline] argparse done, importing torch...", flush=True)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
print("[train_2d_baseline] torch done, importing dataset_2d...", flush=True)

from src.data.dataset_2d import MicroUS2DSliceDataset
print("[train_2d_baseline] dataset_2d done, importing monai_unet_2d...", flush=True)
from src.models.monai_unet_2d import build_monai_unet_2d
print("[train_2d_baseline] all imports done.", flush=True)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    logits: (B, 1, H, W) raw logits (should be on GPU)
    targets: (B, 1, H, W) binary {0,1} (should be on GPU)
    returns: mean Dice over batch (computed on GPU)
    
    All operations stay on the same device as input tensors (no CPU transfers).
    """
    # Ensure all operations stay on GPU - no CPU transfers
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Flatten per-sample (keeps tensors on GPU)
    preds = preds.view(preds.shape[0], -1)
    targets = targets.view(targets.shape[0], -1)

    # All computations on GPU
    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (denom + eps)
    return dice.mean()  # Returns GPU tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate")
    
    parser.add_argument("--splits_dir", type=str, default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--target_h", type=int, default=896)
    parser.add_argument("--target_w", type=int, default=1408)
    parser.add_argument("--transpose_hw", action="store_true")  # default False unless set
    parser.add_argument("--save_ckpt", type=str, default="")    # e.g., runs/baseline.pt

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    target_hw = (args.target_h, args.target_w)

    train_ds = MicroUS2DSliceDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="train",
        target_hw=target_hw,
        transpose_hw=args.transpose_hw,
        only_foreground_slices=False,
    )
    val_ds = MicroUS2DSliceDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        target_hw=target_hw,
        transpose_hw=args.transpose_hw,
        only_foreground_slices=False,
    )

    # Enable pin_memory for faster GPU transfer when using CUDA
    # Note: pin_memory only works effectively with num_workers > 0
    pin_memory = torch.cuda.is_available() and args.num_workers > 0
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=pin_memory  # Faster CPU->GPU transfer (only if num_workers > 0)
    )

    import time
    t0 = time.time()
    print("DEBUG: fetching first train batch...")
    _ = next(iter(train_loader))
    print(f"DEBUG: first train batch fetched in {time.time()-t0:.1f}s")

    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=pin_memory  # Faster CPU->GPU transfer (only if num_workers > 0)
    )

    model = build_monai_unet_2d(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")
    print(f"Target HW: {target_hw} | Batch size: {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")
    print(f"DataLoader: num_workers={args.num_workers}, pin_memory={pin_memory}")
    print("Starting training...")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        print(f"[Epoch {epoch}] Starting training loop...")

        for step, batch in enumerate(train_loader, start=1):
            # Transfer to GPU with non_blocking for async transfer (faster)
            # Note: non_blocking works best with pin_memory=True, but is safe without it
            imgs = batch["image"].to(device, non_blocking=pin_memory)
            lbls = batch["label"].to(device, non_blocking=pin_memory)

            # All forward/backward operations happen on GPU
            optimizer.zero_grad()
            logits = model(imgs)  # Forward pass on GPU
            loss = criterion(logits, lbls)  # Loss computation on GPU
            loss.backward()  # Backward pass on GPU
            optimizer.step()

            # Only convert to float for logging (after GPU computation)
            # Use .item() to properly detach and convert to Python float
            running_loss += loss.item()

            if step == 1 or step % 10 == 0: 
                avg_loss = running_loss / 10
                print(f"[Epoch {epoch}] step {step}/{len(train_loader)} | train_loss={avg_loss:.4f}")
                running_loss = 0.0

        # ---- Val ----
        print(f"[Epoch {epoch}] Starting validation...")
        model.eval()
        dice_sum = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Transfer to GPU with non_blocking for async transfer
                # Note: non_blocking works best with pin_memory=True, but is safe without it
                imgs = batch["image"].to(device, non_blocking=pin_memory)
                lbls = batch["label"].to(device, non_blocking=pin_memory)

                # All evaluation computations on GPU
                logits = model(imgs)  # Forward pass on GPU
                dice = dice_score_from_logits(logits, lbls)  # Dice computation on GPU

                # Only convert to float for accumulation (after GPU computation)
                # Use .item() to properly detach and convert to Python float
                dice_sum += dice.item()
                n_batches += 1

        val_dice = dice_sum / max(n_batches, 1)
        print(f"[Epoch {epoch}] VAL Dice={val_dice:.4f}")

    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        print("Saved checkpoint to:", ckpt_path)


if __name__ == "__main__":
    main()
