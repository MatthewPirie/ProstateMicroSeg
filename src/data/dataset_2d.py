# src/data/dataset_2d.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms_2d import crop_or_pad_2d, pick_random_foreground_center, zscore_normalize


def read_case_ids(txt_path: Path) -> List[str]:
    case_ids: List[str] = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                case_ids.append(line)
    return case_ids


def center_crop_or_pad_2d(arr: np.ndarray, target_hw: Tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    """
    Deterministic center crop/pad (for validation).
    - If larger than target: take centered crop.
    - If smaller than target: symmetric (center) padding.
    """
    assert arr.ndim == 2, f"Expected 2D array, got {arr.ndim}D"
    th, tw = target_hw
    h, w = arr.shape

    # Center crop
    if h > th:
        top = (h - th) // 2
        arr = arr[top : top + th, :]
    if w > tw:
        left = (w - tw) // 2
        arr = arr[:, left : left + tw]

    # Center pad
    h2, w2 = arr.shape
    pad_h = max(th - h2, 0)
    pad_w = max(tw - w2, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h > 0 or pad_w > 0:
        arr = np.pad(
            arr,
            pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    return arr


@dataclass(frozen=True)
class SplitSpec:
    name: str
    case_ids: List[str]
    images_subdir: str
    labels_subdir: str


class MicroUS2DSliceDataset(Dataset):
    """
    One item = one 2D slice (image, label).

    Training behavior (stochastic, nnU-Net-like):
      - Random crop when image > patch
      - Random pad placement when image < patch (image "floats" inside patch)
      - Optional forced foreground sampling: patch centered near random FG pixel when requested
      - Z-score normalization

    Validation behavior (deterministic if deterministic=True):
      - Center crop/pad every time (no randomness, no FG forcing)
    """

    def __init__(
        self,
        dataset_root: str | Path,
        splits_dir: str | Path,
        split: str,  # "train" | "val" | "test"
        target_hw: Tuple[int, int] = (896, 1408),
        transpose_hw: bool = True,
        seed: int = 0,
        fg_center_jitter: int = 32,
        build_fg_index: bool = True,
        fg_threshold: float = 0.5,
        deterministic: bool = False,
    ):
        self.dataset_root = Path(dataset_root)
        self.splits_dir = Path(splits_dir)
        self.split = split.lower()
        self.target_hw = target_hw
        self.transpose_hw = transpose_hw

        self.rng = np.random.default_rng(seed)
        self.fg_center_jitter = int(fg_center_jitter)
        self.fg_threshold = float(fg_threshold)
        self.deterministic = bool(deterministic)

        split_spec = self._make_split_spec(self.split)
        self.case_ids = split_spec.case_ids
        self.images_dir = self.dataset_root / split_spec.images_subdir
        self.labels_dir = self.dataset_root / split_spec.labels_subdir

        # Cache: keep most recently used case in memory
        self._cache_case_id: Optional[str] = None
        self._cache_img: Optional[np.ndarray] = None
        self._cache_lbl: Optional[np.ndarray] = None

        # Index of all (case_id, slice_idx)
        self.index: List[Tuple[str, int]] = self._build_index()

        # Foreground slice indices (used by sampler)
        self.fg_indices: List[int] = []
        if build_fg_index and self.split == "train":
            self.fg_indices = self._build_fg_index()

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

        # Memmap: arrays stored as (S, H, W)
        img = np.load(img_path, mmap_mode="r")
        lbl = np.load(lbl_path, mmap_mode="r")

        self._cache_case_id = case_id
        self._cache_img = img
        self._cache_lbl = lbl
        return img, lbl

    def _build_index(self) -> List[Tuple[str, int]]:
        idx: List[Tuple[str, int]] = []
        for cid in self.case_ids:
            img_path, _ = self._case_paths(cid)
            img = np.load(img_path, mmap_mode="r")
            num_slices = int(img.shape[0])
            for s in range(num_slices):
                idx.append((cid, s))

        if len(idx) == 0:
            raise RuntimeError("Index is empty. Check split files and data paths.")
        return idx

    def _build_fg_index(self) -> List[int]:
        """
        Builds dataset indices whose slice has any foreground in the label.
        Used by the foreground oversampling batch sampler.
        """
        fg: List[int] = []
        for ds_i, (cid, s) in enumerate(self.index):
            _, lbl_path = self._case_paths(cid)
            lbl3d = np.load(lbl_path, mmap_mode="r")  # (S,H,W)
            lbl2d = lbl3d[s, :, :]
            if self.transpose_hw:
                lbl2d = lbl2d.T
            if np.any(lbl2d > self.fg_threshold):
                fg.append(ds_i)
        return fg

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: Union[int, Tuple[int, bool]]) -> Dict[str, Any]:
        img2d = img3d[s, :, :]
        lbl2d = lbl3d[s, :, :]

        if self.transpose_hw:
            img2d = img2d.T
            lbl2d = lbl2d.T

        # Normalize BEFORE crop/pad
        img2d = zscore_normalize(img2d)

        if self.deterministic:
            img2d = center_crop_or_pad_2d(img2d, self.target_hw, pad_value=0.0)
            lbl2d = center_crop_or_pad_2d(lbl2d, self.target_hw, pad_value=0.0)
            force_fg = False
        else:
            center_yx = None
            if force_fg:
                center_yx = pick_random_foreground_center(lbl2d, rng=self.rng, thresh=self.fg_threshold)

            img2d = crop_or_pad_2d(
                img2d, self.target_hw, rng=self.rng,
                pad_value=0.0, center_yx=center_yx, center_jitter=self.fg_center_jitter
            )
            lbl2d = crop_or_pad_2d(
                lbl2d, self.target_hw, rng=self.rng,
                pad_value=0.0, center_yx=center_yx, center_jitter=self.fg_center_jitter
            )

        # Ensure binary label
        lbl2d = (lbl2d > self.fg_threshold).astype(np.float32)

        img_t = torch.from_numpy(img2d).unsqueeze(0).float()
        lbl_t = torch.from_numpy(lbl2d).unsqueeze(0).float()

        return {
            "image": img_t,
            "label": lbl_t,
            "case_id": case_id,
            "slice_idx": torch.tensor(int(s), dtype=torch.int64),
            "force_fg": bool(force_fg),
        }
