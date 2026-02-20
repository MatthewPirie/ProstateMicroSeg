# src/data/dataset_3d_v2.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.normalization import zscore_with_stats
from src.data.transforms_3d_v2 import (
    pick_random_fg_voxel_3d,
    compute_crop_pad_params_3d_v2,
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


class MicroUS3DPatchDatasetV2(Dataset):
    """
    v2 3D patch dataset.

    - One item = one (image_patch, label_patch) sampled from a full 3D volume.
    - Uses memmap-backed npy loading + 1-case cache.
    - Z-score normalization per case using case_stats.json (mean/std).
    - Training sampling:
        * if force_fg=True: pick random FG voxel and place it at a RANDOM location in patch
        * else: pure random crop (or random pad placement if vol < patch)
    - Optional deterministic mode:
        * center_crop_or_pad_3d for both image and label (no force_fg)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        splits_dir: str | Path,
        split: str,  # "train" | "val" | "test"
        target_zyx: Tuple[int, int, int] = (14, 256, 448),
        seed: int = 0,
        fg_threshold: float = 0.5,
        deterministic: bool = False,
        use_case_stats: bool = True,
        case_stats_path: str | Path | None = None,
        eps: float = 1e-8,
    ):
        self.dataset_root = Path(dataset_root)
        self.splits_dir = Path(splits_dir)
        self.split = split.lower()

        self.target_zyx = tuple(int(v) for v in target_zyx)
        self.fg_threshold = float(fg_threshold)
        self.deterministic = bool(deterministic)

        self.use_case_stats = bool(use_case_stats)
        self.eps = float(eps)

        self.rng = np.random.default_rng(int(seed))

        split_spec = self._make_split_spec(self.split)
        self.case_ids = split_spec.case_ids
        self.images_dir = self.dataset_root / split_spec.images_subdir
        self.labels_dir = self.dataset_root / split_spec.labels_subdir

        self.index: List[str] = list(self.case_ids)
        if len(self.index) == 0:
            raise RuntimeError("Index is empty. Check split files and data paths.")

        # Case stats (mean/std per case) for zscore
        self.case_stats: Dict[str, Dict[str, float]] = {}
        if self.use_case_stats:
            if case_stats_path is None:
                # default to dataset_root/case_stats.json
                case_stats_path = self.dataset_root / "case_stats.json"
            case_stats_path = Path(case_stats_path)
            if not case_stats_path.exists():
                raise FileNotFoundError(f"case_stats_path not found: {case_stats_path}")
            with open(case_stats_path, "r") as f:
                self.case_stats = json.load(f)

            missing = [cid for cid in self.index if cid not in self.case_stats]
            if len(missing) > 0:
                ex = ", ".join(missing[:5])
                raise RuntimeError(
                    f"case_stats missing {len(missing)} case_ids for split='{self.split}'. Examples: {ex}"
                )

        # 1-case cache (memmap-backed)
        self._cache_case_id: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_lbl: Optional[np.ndarray] = None

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
        img = np.load(img_path, mmap_mode="r")  # (Z,Y,X)
        lbl = np.load(lbl_path, mmap_mode="r")  # (Z,Y,X)

        self._cache_case_id = case_id
        self._cache_img = img
        self._cache_lbl = lbl
        return img, lbl

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: Union[int, Tuple[int, bool]]) -> Dict[str, Any]:
        # Sampler can pass (index, force_fg)
        if isinstance(i, tuple):
            base_i, force_fg = i
        else:
            base_i, force_fg = i, False

        case_id = self.index[int(base_i)]
        img3d_m, lbl3d_m = self._load_case(case_id)

        img_np = np.asarray(img3d_m)  # (Z,Y,X)
        lbl_np = np.asarray(lbl3d_m)  # (Z,Y,X), binary float32 in your dataset

        # Normalize image (z-score)
        if self.use_case_stats:
            st = self.case_stats[case_id]
            mean = float(st["mean"])
            std = float(st["std"])
        else:
            mean = float(img_np.mean())
            std = float(img_np.std())

        img_np = zscore_with_stats(img_np, mean, std, eps=self.eps)

        # Patch extraction
        if self.deterministic:
            # deterministic validation-style patch
            img_patch = center_crop_or_pad_3d(img_np, self.target_zyx, pad_value=0.0)
            lbl_patch = center_crop_or_pad_3d(lbl_np, self.target_zyx, pad_value=0.0)
            force_fg = False
        else:
            center_zyx = None
            if force_fg:
                center_zyx = pick_random_fg_voxel_3d(lbl_np, rng=self.rng, thresh=self.fg_threshold)
                if center_zyx is None:
                    force_fg = False  # fallback to random sampling if no FG exists

            params = compute_crop_pad_params_3d_v2(
                in_zyx=img_np.shape,
                target_zyx=self.target_zyx,
                rng=self.rng,
                center_zyx=center_zyx,
            )

            img_patch = apply_crop_pad_3d(img_np, params, pad_value=0.0)
            lbl_patch = apply_crop_pad_3d(lbl_np, params, pad_value=0.0)

        # Ensure binary label (safe, given your EDA shows {0,1})
        lbl_patch = (lbl_patch > self.fg_threshold).astype(np.float32)

        # To torch, channel-first: [C,Z,Y,X]
        img_t = torch.from_numpy(img_patch).unsqueeze(0).float()
        lbl_t = torch.from_numpy(lbl_patch).unsqueeze(0).float()

        return {
            "image": img_t,
            "label": lbl_t,
            "case_id": case_id,
            "force_fg": bool(force_fg),
        }
