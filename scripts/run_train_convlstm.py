# ProstateMicroSeg/scripts/run_train_convlstm.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow: from src...

import argparse
import json
import time
import yaml

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset_3d import MicroUS3DDataset
from src.data.dataset_cases import MicroUSCaseDataset
from src.data.samplers_3d import OversampleForegroundBatchSampler3D
from src.data.augmentations_3d import build_train_transforms_3d

from src.models.convlstm_segmentation import build_segmentation_convlstm
from src.train.losses import CompoundBCEDiceLoss

from src.utils.metrics import soft_dice_score, hard_dice_score
from src.utils.visualization import save_val_volume_all_slices_3d_png

from src.train.trainer_convlstm import (
    train_one_epoch_convlstm,
    validate_patches_convlstm,
    validate_cases_convlstm,
)
from src.utils.helper_functions import _get_git_commit, _make_run_dir, save_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # -------------------------
    # Config
    # -------------------------
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to YAML config specifying extractor and augmentations.",
    )

    # -------------------------
    # Data paths
    # -------------------------
    parser.add_argument(
        "--data_root",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/processed/Dataset120_MicroUSProstate",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/home/pirie03/projects/aip-medilab/pirie03/ProstateMicroSeg/dataset/splits",
    )
    parser.add_argument("--case_stats_path", type=str, default="")
    parser.add_argument("--use_case_stats", action="store_true", default=True)
    parser.add_argument("--no_use_case_stats", action="store_false", dest="use_case_stats")

    # -------------------------
    # Run / logging
    # -------------------------
    parser.add_argument("--runs_dir", type=str, default="runs_convlstm")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--save_last", action="store_true", default=True)
    parser.add_argument("--no_save_last", action="store_false", dest="save_last")
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume_ckpt", type=str, default="")

    # -------------------------
    # Training setup
    # -------------------------
    parser.add_argument(
        "--model_variant",
        type=str,
        default="base",
        choices=["small", "base", "large"],
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--steps_per_epoch", type=int, default=250)
    parser.add_argument("--log_every", type=int, default=100)

    # Patch size (ZYX) — Z is the temporal window for the ConvLSTM
    parser.add_argument("--patch_z", type=int, default=16)
    parser.add_argument("--patch_y", type=int, default=256)
    parser.add_argument("--patch_x", type=int, default=448)

    # Foreground forcing
    parser.add_argument("--oversample_fg", type=float, default=0.33)
    parser.add_argument(
        "--oversample_mode",
        type=str,
        default="probabilistic",
        choices=["probabilistic", "deterministic"],
    )
    parser.add_argument("--fg_thr", type=float, default=0.5)

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="polynomial",
        choices=["none", "cosine", "polynomial"],
    )
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--min_lr", type=float, default=1e-6)

    # -------------------------
    # Loss
    # -------------------------
    parser.add_argument("--w_bce", type=float, default=1.0)
    parser.add_argument("--w_dice", type=float, default=1.0)
    parser.add_argument("--batch_dice", action="store_true", default=True)
    parser.add_argument("--no_batch_dice", action="store_false", dest="batch_dice")

    # -------------------------
    # Runtime / dataloader
    # -------------------------
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping. Set <= 0 to disable.",
    )

    # -------------------------
    # Validation
    # -------------------------
    parser.add_argument("--num_val_steps", type=int, default=50)
    parser.add_argument(
        "--fullval_every",
        type=int,
        default=50,
        help="Run full-volume case validation every N epochs. 0=never.",
    )
    parser.add_argument("--val_overlap", type=float, default=0.5)
    parser.add_argument("--val_gaussian", action="store_true", default=True)
    parser.add_argument("--no_val_gaussian", action="store_false", dest="val_gaussian")
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument(
        "--val_carryover",
        action="store_true",
        default=False,
        help="Thread ConvLSTM hidden state across Z windows during case-level validation.",
    )

    return parser


def main() -> None:
    torch.set_num_threads(1)

    parser = build_parser()
    args = parser.parse_args()

    # -------------------------
    # Run directory + metadata
    # -------------------------
    run_dir = _make_run_dir(args.runs_dir, args.run_name)
    tb_dir = run_dir / "tb"
    writer = SummaryWriter(log_dir=str(tb_dir))

    print("Run dir:", run_dir, flush=True)

    with open(run_dir / "git_commit.txt", "w") as f:
        f.write(_get_git_commit() + "\n")

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    pin_memory = torch.cuda.is_available() and args.num_workers > 0
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(torch.cuda.is_available() and bool(args.amp)),
    )

    # -------------------------
    # Load training config (YAML)
    # -------------------------
    train_config_path = Path(args.train_config)
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config not found: {train_config_path}")

    with open(train_config_path, "r") as f:
        train_cfg = yaml.safe_load(f)

    if train_cfg is None:
        train_cfg = {}

    extractor_cfg = train_cfg.get("extractor", {})
    aug_cfg = train_cfg.get("augmentations", {})

    extractor_name = str(extractor_cfg.get("name", "full_inplane_zwindow"))
    extractor_kwargs = dict(extractor_cfg.get("kwargs", {}))
    enabled_augs = list(aug_cfg.get("enabled", []))

    print(f"Train config: {train_config_path}", flush=True)
    print(f"Extractor: {extractor_name}", flush=True)
    print(f"Extractor kwargs: {extractor_kwargs}", flush=True)
    print(f"Enabled augmentations: {enabled_augs}", flush=True)

    # -------------------------
    # Derived
    # -------------------------
    patch_zyx = (int(args.patch_z), int(args.patch_y), int(args.patch_x))
    steps_per_epoch = int(args.steps_per_epoch)
    total_steps = int(args.epochs) * steps_per_epoch

    case_stats_path = args.case_stats_path.strip()
    if case_stats_path == "":
        case_stats_path = str(Path(args.data_root) / "case_stats.json")

    train_tf = build_train_transforms_3d(
        patch_zyx=patch_zyx,
        enabled_augs=enabled_augs,
    )

    # -------------------------
    # Datasets
    # -------------------------
    train_ds = MicroUS3DDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="train",
        target_zyx=patch_zyx,
        seed=0,
        fg_threshold=float(args.fg_thr),
        deterministic=False,
        transform=train_tf,
        use_case_stats=bool(args.use_case_stats),
        case_stats_path=case_stats_path,
        extractor_name=extractor_name,
        extractor_kwargs=extractor_kwargs,
    )

    val_patch_ds = MicroUS3DDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        target_zyx=patch_zyx,
        seed=0,
        fg_threshold=float(args.fg_thr),
        deterministic=False,
        transform=None,
        use_case_stats=bool(args.use_case_stats),
        case_stats_path=case_stats_path,
        extractor_name=extractor_name,
        extractor_kwargs=extractor_kwargs,
    )

    val_case_ds = MicroUSCaseDataset(
        dataset_root=args.data_root,
        splits_dir=args.splits_dir,
        split="val",
        fg_threshold=float(args.fg_thr),
        use_case_stats=bool(args.use_case_stats),
        case_stats_path=case_stats_path if bool(args.use_case_stats) else None,
    )

    # -------------------------
    # Samplers + loaders
    # -------------------------
    dl_kwargs = {}
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 2

    train_sampler = OversampleForegroundBatchSampler3D(
        dataset=train_ds,
        batch_size=int(args.batch_size),
        oversample_foreground_percent=float(args.oversample_fg),
        seed=0,
        drop_last=True,
        mode=str(args.oversample_mode),
        shuffle=True,
    )

    val_patch_sampler = OversampleForegroundBatchSampler3D(
        dataset=val_patch_ds,
        batch_size=int(args.batch_size),
        oversample_foreground_percent=float(args.oversample_fg),
        seed=1,
        drop_last=True,
        mode=str(args.oversample_mode),
        shuffle=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    val_patch_loader = DataLoader(
        val_patch_ds,
        batch_sampler=val_patch_sampler,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    val_case_loader = DataLoader(
        val_case_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        **dl_kwargs,
    )

    print(f"Train cases: {len(train_ds)} | Val cases: {len(val_case_ds)}", flush=True)
    print(f"Patch ZYX: {patch_zyx} | batch={args.batch_size} steps/epoch={steps_per_epoch}", flush=True)
    print(f"Oversample FG: p={args.oversample_fg} mode={args.oversample_mode}", flush=True)
    print(
        f"Online val: num_val_steps={args.num_val_steps} | "
        f"Fullval: every={args.fullval_every} | "
        f"val_carryover={args.val_carryover}",
        flush=True,
    )

    # -------------------------
    # Model
    # -------------------------
    model, model_meta = build_segmentation_convlstm(
        in_channels=1,
        out_channels=1,
        variant=args.model_variant,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SegmentationConvLSTM variant={args.model_variant} | params={total_params:,}", flush=True)

    # -------------------------
    # Loss
    # -------------------------
    criterion = CompoundBCEDiceLoss(
        w_bce=float(args.w_bce),
        w_dice=float(args.w_dice),
        batch_dice=bool(args.batch_dice),
    ).to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(args.lr),
            momentum=float(args.momentum),
            nesterov=True,
            weight_decay=float(args.weight_decay),
        )

    # -------------------------
    # Scheduler (iteration-based, stepped inside trainer)
    # -------------------------
    scheduler = None
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=float(args.min_lr),
        )
    elif args.lr_scheduler == "polynomial":
        power = float(args.poly_power)

        def _poly(step: int) -> float:
            s = min(step, max(total_steps, 1))
            return (1.0 - s / max(total_steps, 1)) ** power

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_poly)

    # -------------------------
    # Save config
    # -------------------------
    run_config = {
        "cli_args": vars(args),
        "train_config": train_cfg,
        "model": model_meta,
        "patch_zyx": patch_zyx,
        "case_stats_path": case_stats_path,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # -------------------------
    # Resume
    # -------------------------
    start_epoch = 1
    best_valp = float("-inf")

    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1

        extra = ckpt.get("extra", {})
        if isinstance(extra, dict) and "best_valp_dice_thr05" in extra:
            best_valp = float(extra["best_valp_dice_thr05"])

        print(f"Resumed from: {ckpt_path}", flush=True)
        print(f"Starting at epoch: {start_epoch}", flush=True)
        print(f"Current best_valp_dice_thr05: {best_valp}", flush=True)

    # -------------------------
    # Train loop
    # -------------------------
    metrics_path = run_dir / "metrics.jsonl"

    for epoch in range(start_epoch, start_epoch + int(args.epochs)):
        t0 = time.time()

        train_metrics = train_one_epoch_convlstm(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            log_every=int(args.log_every),
            pin_memory=pin_memory,
            scheduler=scheduler,
            scaler=scaler,
            amp=bool(args.amp),
            grad_clip_norm=float(args.grad_clip_norm),
        )
        t1 = time.time()

        valp_metrics = validate_patches_convlstm(
            model=model,
            val_loader=val_patch_loader,
            criterion=criterion,
            device=device,
            dice_soft_fn=lambda lg, y: soft_dice_score(lg, y, batch_dice=bool(args.batch_dice)),
            dice_hard_fn=hard_dice_score,
            epoch=epoch,
            num_val_steps=int(args.num_val_steps),
            pin_memory=pin_memory,
        )
        t2 = time.time()

        fullval_metrics = None
        do_full = int(args.fullval_every) > 0 and (epoch % int(args.fullval_every) == 0)

        if do_full:
            fullval_metrics = validate_cases_convlstm(
                model=model,
                val_loader=val_case_loader,
                criterion=criterion,
                device=device,
                dice_soft_fn=lambda lg, y: soft_dice_score(lg, y, batch_dice=bool(args.batch_dice)),
                dice_hard_fn=hard_dice_score,
                epoch=epoch,
                roi_size=patch_zyx,
                overlap=float(args.val_overlap),
                sw_batch_size=int(args.sw_batch_size),
                gaussian=bool(args.val_gaussian),
                use_hidden_carryover=bool(args.val_carryover),
                pin_memory=pin_memory,
            )
        t3 = time.time()

        # TensorBoard
        writer.add_scalar("train/total_loss", train_metrics["train_total_loss"], epoch)
        writer.add_scalar("train/bce", train_metrics["train_bce"], epoch)
        writer.add_scalar("train/dice_loss", train_metrics["train_dice_loss"], epoch)

        writer.add_scalar("val_patch/total_loss", valp_metrics["valp_total_loss"], epoch)
        writer.add_scalar("val_patch/dice_thr05", valp_metrics["valp_dice_thr05"], epoch)
        writer.add_scalar("val_patch/dice_soft", valp_metrics["valp_dice_soft"], epoch)

        if fullval_metrics is not None:
            writer.add_scalar("val_case/total_loss", fullval_metrics["val_total_loss"], epoch)
            writer.add_scalar("val_case/dice_soft", fullval_metrics["val_dice_soft"], epoch)
            writer.add_scalar("val_case/dice_thr05", fullval_metrics["val_dice_thr05"], epoch)

        writer.flush()

        print(
            f"[Epoch {epoch}] time_sec={(t3 - t0):.1f} "
            f"train={(t1 - t0):.1f} val_patch={(t2 - t1):.1f} "
            f"val_case={(t3 - t2):.1f}",
            flush=True,
        )

        record = {
            "epoch": epoch,
            "epoch_sec": float(t3 - t0),
            **train_metrics,
            **valp_metrics,
        }
        if fullval_metrics is not None:
            record.update({f"full_{k}": v for k, v in fullval_metrics.items()})

        with open(metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        checkpoint_extra = {
            "best_valp_dice_thr05": best_valp,
            "train_metrics": train_metrics,
            "valp_metrics": valp_metrics,
            "fullval_metrics": fullval_metrics if fullval_metrics is not None else {},
            "cli_args": vars(args),
            "train_config": train_cfg,
        }

        if bool(args.save_last):
            save_checkpoint(
                ckpt_path=run_dir / "checkpoint_last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra=checkpoint_extra,
            )

        if bool(args.save_best) and float(valp_metrics["valp_dice_thr05"]) > best_valp:
            best_valp = float(valp_metrics["valp_dice_thr05"])
            checkpoint_extra["best_valp_dice_thr05"] = best_valp

            save_checkpoint(
                ckpt_path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra=checkpoint_extra,
            )
            print(
                f"[Epoch {epoch}] New best VAL_PATCH Dice(thr=0.5)={best_valp:.4f} -> saved checkpoint_best.pt",
                flush=True,
            )

        if int(args.save_every) > 0 and (epoch % int(args.save_every) == 0):
            save_checkpoint(
                ckpt_path=run_dir / f"checkpoint_epoch{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                extra=checkpoint_extra,
            )

    # -------------------------
    # Final visualization (best checkpoint)
    # -------------------------
    best_ckpt_path = run_dir / "checkpoint_best.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint for visualization: {best_ckpt_path}", flush=True)
    else:
        print("No best checkpoint found, visualizing with last epoch weights.", flush=True)

    print("Running final visualization on validation case...", flush=True)
    model.eval()

    vis_item = val_case_ds[0]
    vis_img = vis_item["image"].unsqueeze(0).to(device)
    vis_lbl = vis_item["label"].unsqueeze(0).to(device)
    vis_case_id = vis_item["case_id"]

    vis_dir = run_dir / "vis"
    out_png = vis_dir / f"final_{vis_case_id}_all_slices.png"

    save_val_volume_all_slices_3d_png(
        model=model,
        img=vis_img,
        lbl=vis_lbl,
        out_png=out_png,
        roi_size=patch_zyx,
        vis_mode="fullinplane",
        thr=0.5,
        overlap=float(args.val_overlap),
        sw_batch_size=int(args.sw_batch_size),
        gaussian=bool(args.val_gaussian),
        amp=bool(args.amp),
    )

    print(f"Saved visualization: {out_png}", flush=True)
    print("Training complete.", flush=True)

    writer.close()


if __name__ == "__main__":
    main()
