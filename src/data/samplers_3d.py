# src/data/samplers_3d.py

from __future__ import annotations

from typing import Iterator, List, Tuple
import math
import numpy as np
from torch.utils.data import Sampler


class OversampleForegroundBatchSampler3D(Sampler[List[Tuple[int, bool]]]):
    """
    Yields batches of (index, force_fg) pairs for 3D volumes.

    - Indices are always drawn from all cases WITH replacement (nnU-Net style).
    - k samples per batch are tagged force_fg=True, meaning the dataset will pick
      a foreground voxel as the patch center for those samples.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        oversample_foreground_percent: float = 0.33,
        seed: int = 0,
        drop_last: bool = True,
        ensure_at_least_one_fg: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.p = float(oversample_foreground_percent)
        self.drop_last = bool(drop_last)
        self.ensure_at_least_one_fg = bool(ensure_at_least_one_fg)

        self.rng = np.random.default_rng(seed)
        self.all_indices = np.arange(len(dataset), dtype=np.int64)

    def __iter__(self) -> Iterator[List[Tuple[int, bool]]]:
        n = len(self.all_indices)
        num_batches = (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)

        for _ in range(num_batches):
            k = int(round(self.batch_size * self.p))
            k = max(0, min(k, self.batch_size))
            if self.ensure_at_least_one_fg:
                k = max(1, k)

            rest_n = self.batch_size - k

            rest = self.rng.choice(self.all_indices, size=rest_n, replace=True)
            fg = self.rng.choice(self.all_indices, size=k, replace=True)

            batch = [(int(i), False) for i in rest] + [(int(i), True) for i in fg]
            yield batch

    def __len__(self) -> int:
        n = len(self.all_indices)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
