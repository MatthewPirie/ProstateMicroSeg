# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**ProstateMicroSeg** is a deep learning project for segmenting the prostate gland in micro-ultrasound volumes. The task is **volumetric binary segmentation**: given a 3D micro-ultrasound scan (Z axial slices × H × W), predict a binary mask on every slice indicating prostate tissue.

Three parallel pipelines are under development and comparison:

| Pipeline | Model | Input | Output |
|----------|-------|-------|--------|
| **2D** | MONAI UNet 2D | `(B, 1, H, W)` single slice | `(B, 1, H, W)` |
| **3D** | MONAI DynUNet (nnU-Net style) | `(B, 1, Z, H, W)` patch | `(B, 1, Z, H, W)` |
| **ConvLSTM** | SegmentationConvLSTM | `(B, 1, Z, H, W)` window | `(B, 1, Z, H, W)` |

> **Related project:** `NeedleMicroSeg` at `/home/pirie03/projects/aip-medilab/pirie03/NeedleMicroSeg` — a parallel needle segmentation project with similar design patterns but a different task (single annotated frame, not full volume).

## Environment

```bash
source /home/pirie03/envs/prostate_microseg/bin/activate
```

No `requirements.txt`. Key dependencies: `torch`, `monai`, `numpy`, `SimpleITK`, `tensorboard`, `pyyaml`.

There is no `__init__.py` at repo root — training scripts add the repo root to `sys.path` via `sys.path.insert(0, ...)`. Run all scripts from the repo root.

## Data

**Processed dataset:** `dataset/processed/Dataset120_MicroUSProstate/`

```
dataset/
  processed/Dataset120_MicroUSProstate/
    imagesTr/          # train + val volumes as .npy
    labelsTr/          # train + val labels as .npy
    imagesTs/          # test volumes as .npy
    labelsTs/          # test labels as .npy
    case_stats.json    # per-case mean/std for z-score normalization
  splits/
    train.txt          # 45 case IDs
    val.txt            # 10 case IDs
    test.txt           # 20 case IDs
  raw/                 # original NIfTI files (not used during training)
```

Each `.npy` file is a `float32` array with shape `(Z, Y, X)` (no channel dim). Labels are binary `{0, 1}`. Z is the anisotropic (axial/slice) dimension; Y×X is the in-plane image. Spatial resolution is anisotropic — Z spacing is coarser than Y/X — which is why the 3D model uses `(1,3,3)` kernels at early stages.

**One-time data prep** (run before first training):
```bash
python scripts/preconvert_nii_to_npy.py   # NIfTI → .npy
python scripts/make_case_stats.py          # compute per-case mean/std → case_stats.json
```

## Code Structure

### Data (`src/data/`)

**Datasets:**
- `dataset_2d.py` — `MicroUS2DSliceDataset`: one item = one 2D slice. Loads full volumes with a 1-case LRU cache, z-score normalizes, then crops/pads each slice to `target_hw`. Training: stochastic crop with optional foreground forcing. Validation: deterministic center crop/pad.
- `dataset_3d.py` — `MicroUS3DDataset`: one item = one 3D patch/window extracted by a pluggable extractor. Normalizes using `case_stats.json`, runs the extractor, returns `(1, Z, Y, X)` float32 tensor.
- `dataset_cases.py` — `MicroUSCaseDataset`: one item = one full 3D volume `(1, Z, Y, X)`. Used for case-level validation and visualization by all three pipelines.

**Extractors (`src/data/extractors_3d/`):**

Registered in `__init__.py::get_extractor(name)`. All return `(img_out, lbl_out, meta)`.

- `patch_centered_fg` — fixed-size ZYX patch centered on a random foreground voxel with optional jitter (`center_jitter_zyx`)
- `patch_random_fg` — fixed-size ZYX patch with random placement biased towards foreground
- `full_inplane_zstrided` — full in-plane (no XY cropping), evenly-spaced Z strides
- `full_inplane_zwindow` — full in-plane, contiguous Z window of length `z_window`; pads with zeros if volume is shorter than `z_window`. Used by the ConvLSTM pipeline. See `full_inplane_zwindow.py` for details on stochastic/deterministic/force-fg modes.

**Samplers:**
- `samplers_2d.py` — `OversampleForegroundBatchSampler`: ensures ≥33% of slices per batch contain foreground. Critical for the class-imbalanced prostate task.
- `samplers_3d.py` — `OversampleForegroundBatchSampler3D`: same idea for 3D patches; supports `probabilistic` and `deterministic` modes.

**Augmentations:**
- `augmentations_2d.py` — `build_train_transforms_2d(target_hw, enabled_augs)`: MONAI-based; options `{"flip","affine","elastic","noise","smooth","shift_intensity","scale_intensity","contrast"}`
- `augmentations_3d.py` — `build_train_transforms_3d(patch_zyx, enabled_augs)`: MONAI-based; options `{"rotscale","translate","noise","brightness","contrast"}`; spatial transforms are conservative to respect anisotropy

### Models (`src/models/`)

- `monai_unet_2d.py` — `build_monai_unet_2d(in_channels, out_channels, variant)`: wraps MONAI `UNet`; variants `"tiny"`, `"small"`, `"base"`
- `monai_unet_3d.py` — `build_monai_unet_3d(in_channels, out_channels, variant)`: wraps MONAI `DynUNet` (falls back to `UNet`); anisotropic kernels/strides matching nnU-Net 3D_fullres; variants:
  - `"nnunet_fullres"` — 7 stages, filters `[32,64,128,256,320,320,320]`, early `(1,3,3)` kernels
  - `"small"` — 5 stages, filters `[32,64,128,256,320]`
- `convlstm.py` — `ConvLSTM`: multi-layer spatiotemporal recurrence, input `(B, T, C, H, W)` batch-first
- `convlstm_segmentation.py` — `SegmentationConvLSTM`: 2D encoder (frame-by-frame via batch flattening) → ConvLSTM bottleneck → 2D decoder with skip connections; outputs `(B, 1, Z, H, W)`; variants `"small"`, `"base"`, `"large"` (encoder channels `(32,64,128)` / `(32,64,128,256)` / `(32,64,128,256,512)`)

### Training (`src/train/`)

- `losses.py` — `CompoundBCEDiceLoss(w_bce, w_dice, batch_dice)`: weighted BCE + soft Dice
- `trainer_2d.py` — `train_one_epoch_2d`, `validate_case_level_2d`: case-level val slices the full volume and batches slices through the 2D model
- `trainer_3d.py` — `train_one_epoch`, `validate_patches_3d`, `validate_cases_slidingwindow_3d`, `validate_cases_fullinplane_3d`: full-volume val uses MONAI sliding-window inference; `fullinplane` variant is used when the extractor is `full_inplane_zwindow`
- `trainer_convlstm.py` — `train_one_epoch_convlstm`, `validate_patches_convlstm`, `validate_cases_convlstm`: adds gradient clipping; case-level val supports two modes — MONAI sliding-window (`use_hidden_carryover=False`) or non-overlapping windows with threaded LSTM state (`use_hidden_carryover=True`); volumes shorter than `z_window` are passed directly to the model without padding

All trainers use **nnU-Net-style fixed steps per epoch** (iterator restarted on exhaustion), AMP, and cosine/polynomial LR scheduling stepped per iteration.

### Utils (`src/utils/`)

- `normalization.py` — `zscore_with_stats(arr, mean, std)`: applies precomputed per-case statistics from `case_stats.json`
- `metrics.py` — `soft_dice_score`, `hard_dice_score`
- `helper_functions.py` — `_make_run_dir`, `save_checkpoint`, `_get_git_commit`
- `visualization.py` — `save_val_volume_all_slices_2d_png`, `save_val_volume_all_slices_3d_png`: saves a grid of all slices (image / GT / prediction) for one validation case

## Training Scripts

All scripts live in `scripts/`. Run from repo root.

**2D:**
```bash
python scripts/run_train_2d.py \
    --train_config configs/train_2d/all_augs.yaml \
    --runs_dir runs/a_runs_2d \
    --epochs 10 --batch_size 8 --steps_per_epoch 588 \
    --optimizer adam --lr 3e-4 --lr_scheduler cosine \
    --model_variant base
```

**3D:**
```bash
python scripts/run_train_3d.py \
    --train_config configs/train_3d/fullinplane_zwindow_translate.yaml \
    --runs_dir runs/b_runs_3d \
    --epochs 10 --batch_size 2 --steps_per_epoch 200 \
    --model_variant nnunet_fullres \
    --patch_z 14 --patch_y 384 --patch_x 384 \
    --optimizer adam --lr 1e-4 --lr_scheduler polynomial
```

**ConvLSTM:**
```bash
python scripts/run_train_convlstm.py \
    --train_config configs/train_convlstm/fullinplane_zwindow.yaml \
    --runs_dir runs/c_runs_convlstm \
    --epochs 10 --batch_size 4 --steps_per_epoch 150 \
    --model_variant base \
    --patch_z 45 --patch_y 256 --patch_x 256 \
    --optimizer adam --lr 1e-4 \
    --fullval_every 9 --val_carryover
```

**SLURM** (preferred for longer runs):
```bash
cd scripts/slurm
sbatch a_run_train_2d.sh
sbatch b_run_train_3d.sh
sbatch c_run_train_convlstm.sh
```

SLURM logs go to `scripts/slurm/a_logs/`. Output runs go to `runs/a_runs_2d/`, `runs/b_runs_3d/`, `runs/c_runs_convlstm/`.

## Config Format

Configs live in `configs/train_2d/`, `configs/train_3d/`, `configs/train_convlstm/`. They control the extractor (3D only) and which augmentations are enabled. All other hyperparameters are CLI args.

**2D config** (augmentations only):
```yaml
augmentations:
  enabled:
    - flip
    - affine
    - elastic
    - noise
    - smooth
    - shift_intensity
    - scale_intensity
    - contrast
```

**3D / ConvLSTM config** (extractor + augmentations):
```yaml
extractor:
  name: full_inplane_zwindow   # or patch_centered_fg / patch_random_fg / full_inplane_zstrided
  kwargs: {}                   # extractor-specific overrides, e.g. center_jitter_zyx: [2,32,32]

augmentations:
  enabled:
    - translate                # conservative; rotscale / noise / brightness / contrast also available
```

## Output Structure

Each run produces:
```
runs/<prefix>/<run_name>/
  checkpoint_last.pt     # weights after final epoch
  checkpoint_best.pt     # best weights by patch-val Dice (thr=0.5)
  config.json            # full config snapshot (CLI args + YAML + model meta)
  metrics.jsonl          # one JSON line per epoch
  git_commit.txt
  tb/                    # TensorBoard logs
  vis/                   # PNG grid: all slices of one val case at end of training
```

TensorBoard:
```bash
tensorboard --logdir runs/
```
