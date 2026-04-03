# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ProstateMicroSeg** is a deep learning segmentation pipeline for prostate micro-ultrasound (US) images, supporting both 2D and 3D approaches. The current focus is on 3D pipelines using a MONAI UNet architecture matching the nnU-Net 3D_fullres configuration.

## Environment

The Python environment is at `/home/pirie03/envs/prostate_microseg/`. Activate with:
```bash
source /home/pirie03/envs/prostate_microseg/bin/activate
```

Key dependencies (no requirements.txt): PyTorch, MONAI, NumPy, SimpleITK, TensorBoard.

## Data Preparation

Run these once before training:

```bash
# 1. Convert NIfTI → NumPy (faster I/O during training)
python scripts/preconvert_nii_to_npy.py

# 2. Compute per-case normalization statistics
python scripts/make_case_stats.py
```

Raw data lives in `dataset/raw/Dataset120_MicroUSProstate/`, preprocessed data in `dataset/processed/`.

## Running Training

**Local (direct execution):**
```bash
# 3D pipeline v2 (hardcoded config)
python scripts/run_train_3d_v2.py

# 3D pipeline v3 (YAML config-driven, preferred)
python scripts/run_train_3d_v3.py --config configs/train_3d/fullinplane_zwindow_translate.yaml
```

**On HPC via SLURM:**
```bash
sbatch scripts/slurm/run_train_adam_full_inplane.sh  # latest config
sbatch scripts/slurm/run_train_3d_adam_v2.sh
```

SLURM logs go to `scripts/slurm/logs_v2/`. Training outputs (checkpoints, TensorBoard logs) go to `runs_3d_v2/`.

## Architecture

### Pipeline Versions

There are multiple generations of the pipeline (v1 → v2 → v3). The v3 scripts (`run_train_3d_v3.py`, `trainer_3d_v2.py`, `dataset_3d_v2.py`) are the current ones. v1 files are kept for reference.

### Pluggable Extractor Pattern

The most important design pattern is the **extractor registry** in `src/data/extractors/`. During training, each dataset item involves:
1. Loading a full case volume (with 1-case LRU cache)
2. Applying z-score normalization using precomputed stats (`src/utils/normalization.py`)
3. Extracting a patch/window via a pluggable extractor
4. Applying MONAI augmentations
5. Returning a `[C, Z, Y, X]` float32 tensor

Extractors are selected by name via YAML config:
- `patch_centered_fg` — fixed-size patch centered on foreground voxel with optional jitter
- `patch_random_fg` — random patch with foreground-biased sampling
- `full_inplane_zstrided` — full XY with strided Z-axis windows
- `full_inplane_zwindow` — full XY resolution with contiguous Z window (current default); supports stochastic FG-forced sampling

New extractors must be registered in `src/data/extractors/__init__.py`.

### Foreground Oversampling

`src/data/samplers_3d_v2.py::OversampleForegroundBatchSampler3DV2` enforces a minimum foreground proportion per batch (e.g., 33%). This is critical for the heavily class-imbalanced prostate segmentation task.

### Model

`src/models/monai_unet_3d.py` — MONAI UNet with parameters matching nnU-Net 3D_fullres:
- 7 stages, filters `[32, 64, 128, 256, 320, 320, 320]`
- Anisotropic kernels/strides to handle micro-US voxel spacing
- Variants: `"nnunet_fullres"` and `"small"`

### Loss

`src/train/losses.py::CompoundBCEDiceLoss` — weighted sum of BCE and soft Dice, averaged batch-wise.

### Training Loop

`src/train/trainer_3d_v2.py` — nnU-Net-style fixed steps-per-epoch (not full dataset epochs), AMP support, cosine/polynomial LR scheduling, TensorBoard logging.

### YAML Config Schema

```yaml
extractor:
  name: full_inplane_zwindow   # extractor registry key
  z_window: 16                 # extractor-specific params
  ...
augmentations:
  translate: true              # enable/disable individual augmentations
  rotate: false
  ...
```

## Visualization & Debugging

```bash
# Visualize 3D predictions
python scripts/tests/vis_pred_3d.py
python scripts/tests/vis_pred_3d_patch.py

# Compare extractors interactively
jupyter notebook scripts/tests/compare_extractors_3d.ipynb
```

TensorBoard:
```bash
tensorboard --logdir runs_3d_v2/
```
