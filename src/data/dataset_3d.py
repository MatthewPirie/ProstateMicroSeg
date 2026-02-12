# src/data/dataset_3d.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms_3d import (
    zscore_normalize_3d,
    pick_random_foreground_center_3d,
    compute_crop_pad_params_3d,
    apply_crop_pad_3d,
    center_crop_or_pad_3d,
)


def read_case_ids(txt_path: Path) -> List[str]:
    case_ids: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                case_ids.append(line)
    return case_ids


@dataclass(frozen=True)
class SplitSpec:
    name: str
    case_ids: List[str]
    images_subdir: str
    labels_subdir: str


class MicroUS3DVolumeDataset(Dataset):
    """
    One item = one 3D patch sampled from a volume.

    Training behavior (stochastic, nnU-Net-like):
      - Random crop when vol > patch
      - Random pad placement when vol < patch (vol "floats" inside patch)
      - Optional forced foreground sampling: patch centered near random FG voxel when requested
      - Z-score normalization

    Validation behavior (deterministic if deterministic=True):
      - Center crop/pad every time (no randomness, no FG forcing)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        splits_dir: str | Path,
        split: str,  # "train" | "val" | "test"
        target_zyx: Tuple[int, int, int] = (32, 256, 256),
        seed: int = 0,
        fg_center_jitter_zyx: Tuple[int, int, int] = (2, 32, 32),
        fg_threshold: float = 0.5,
        deterministic: bool = False,
        do_augment: bool = False,   # hook, wire later
        augment_seed: int = 0,      # hook, wire later
        case_stats_path: str | Path | None = None,   # NEW
    ):
        self.dataset_root = Path(dataset_root)
        self.splits_dir = Path(splits_dir)
        case_stats_path = self.dataset_root / "case_stats.json"
        with open(case_stats_path, "r") as f:
            self.case_stats = json.load(f)

        self.split = split.lower()
        self.target_zyx = tuple(int(v) for v in target_zyx)

        self.rng = np.random.default_rng(seed)
        self.fg_center_jitter_zyx = tuple(int(v) for v in fg_center_jitter_zyx)
        self.fg_threshold = float(fg_threshold)
        self.deterministic = bool(deterministic)

        # Augmentations: train-only hook (keep None for now)
        self.do_augment = bool(do_augment) and (self.split == "train") and (not self.deterministic)
        self.aug = None
        _ = int(augment_seed)  # reserved for later wiring

        split_spec = self._make_split_spec(self.split)
        self.case_ids = split_spec.case_ids
        self.images_dir = self.dataset_root / split_spec.images_subdir
        self.labels_dir = self.dataset_root / split_spec.labels_subdir

        # Cache: keep most recently used case in memory (memmap-backed)
        self._cache_case_id: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_lbl: Optional[np.ndarray] = None

        # Index of cases (one item per volume). Patch sampling happens inside __getitem__.
        self.index: List[str] = list(self.case_ids)
        if len(self.index) == 0:
            raise RuntimeError("Index is empty. Check split files and data paths.")

    def _make_split_spec(self, split: str) -> SplitSpec:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        case_ids = read_case_ids(self.splits_dir / f"{split}.txt")
        if split == "test":
            return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTs", labels_subdir="labelsTs")
        return SplitSpec(name=split, case_ids=case_ids, images_subdir="imagesTr", labels_subdir="labelsTr")

    def _case_paths(self, case_id: str) -> Tuple[Path, Path]:
        img_path = self.images_dir / f"{case_id}.npy"
        lbl_path = self.labels_dir / f"{case_id}.npy"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not lbl_path.exists():
            raise FileNotFoundError(f"Missing label: {lbl_path}")
        return img_path, lbl_path

    def _load_case(self, case_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if self._cache_case_id == case_id and self._cache_img is not None and self._cache_lbl is not None:
            return self._cache_img, self._cache_lbl

        img_path, lbl_path = self._case_paths(case_id)

        # Expect stored as (Z, Y, X)
        img = np.load(img_path, mmap_mode="r")
        lbl = np.load(lbl_path, mmap_mode="r")

        self._cache_case_id = case_id
        self._cache_img = img
        self._cache_lbl = lbl
        return img, lbl

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: Union[int, Tuple[int, bool]]) -> Dict[str, Any]:
        # Allow sampler to pass (index, force_fg)
        if isinstance(i, tuple):
            base_i, force_fg = i
        else:
            base_i, force_fg = i, False

        case_id = self.index[int(base_i)]
        img3d, lbl3d = self._load_case(case_id)

        # if (not hasattr(self, "_printed_shape")) or (self._printed_shape is False):
        #     print(
        #         f"[{self.split}] example case {case_id}: img3d shape={img3d.shape}, lbl3d shape={lbl3d.shape}",
        #         flush=True,
        #     )
        #     self._printed_shape = True

        # Convert memmap to ndarray view where needed (still cheap) and normalize BEFORE crop/pad
        img3d_np = zscore_normalize_3d(np.asarray(img3d))
        lbl3d_np = np.asarray(lbl3d)

        if self.deterministic:
            # Validation: deterministic center crop/pad, no FG forcing
            img_patch = center_crop_or_pad_3d(img3d_np, self.target_zyx, pad_value=0.0)
            lbl_patch = center_crop_or_pad_3d(lbl3d_np, self.target_zyx, pad_value=0.0)
            force_fg = False
        else:
            # Training: stochastic crop/pad, optional FG centering
            center_zyx = None
            if force_fg:
                center_zyx = pick_random_foreground_center_3d(lbl3d_np, rng=self.rng, thresh=self.fg_threshold)

            params = compute_crop_pad_params_3d(
                in_zyx=img3d_np.shape,
                target_zyx=self.target_zyx,
                rng=self.rng,
                center_zyx=center_zyx,
                center_jitter_zyx=self.fg_center_jitter_zyx,
            )

            # Apply SAME plan to both image and label
            img_patch = apply_crop_pad_3d(img3d_np, params, pad_value=0.0)
            lbl_patch = apply_crop_pad_3d(lbl3d_np, params, pad_value=0.0)

        # Ensure binary label (before any aug)
        lbl_patch = (lbl_patch > self.fg_threshold).astype(np.float32)

        # Convert to torch tensors, add channel dim: [C, Z, Y, X]
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        lbl_t = torch.from_numpy(lbl_patch).unsqueeze(0).float()

        # Optional MONAI aug hook (wire later)
        if self.aug is not None:
            out = self.aug({"image": img_t, "label": lbl_t})
            img_t = out["image"]
            lbl_t = out["label"]

        return {
            "image": img_t,
            "label": lbl_t,
            "case_id": case_id,
            "force_fg": bool(force_fg),
        }
