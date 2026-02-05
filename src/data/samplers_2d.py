# src/data/samplers_2d.py

from __future__ import annotations

from typing import Iterator, List, Tuple
import math
import numpy as np
from torch.utils.data import Sampler


class OversampleForegroundBatchSampler(Sampler[List[Tuple[int, bool]]]):
    """
    Yields batches of (index, force_fg) pairs.

    Behavior:
      - Each batch has k = round(batch_size * oversample_foreground_percent) forced-FG samples.
      - Forced-FG indices are drawn from dataset.fg_indices WITH replacement (so FG can repeat).
      - Remaining indices are drawn from all indices WITH replacement.
      - The forced-FG samples are placed at the end of the batch (nnU-Net style).

    Epoch length:
      - Determined by __len__() (usually floor(N / batch_size) if drop_last=True).
      - Does NOT guarantee each slice appears once per epoch. That is intentional.
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
        self.fg_indices = np.array(getattr(dataset, "fg_indices", []), dtype=np.int64)

        if self.ensure_at_least_one_fg and self.fg_indices.size == 0:
            raise RuntimeError("No foreground slices found in dataset.fg_indices, cannot oversample foreground.")

    def __iter__(self) -> Iterator[List[Tuple[int, bool]]]:
        n = len(self.all_indices)
        if self.drop_last:
            num_batches = n // self.batch_size
        else:
            num_batches = math.ceil(n / self.batch_size)

        for _ in range(num_batches):
            k = int(round(self.batch_size * self.p))
            k = max(0, min(k, self.batch_size))

            if self.ensure_at_least_one_fg and self.fg_indices.size > 0:
                k = max(1, k)

            if k > 0 and self.fg_indices.size > 0:
                fg = self.rng.choice(self.fg_indices, size=k, replace=True)
            else:
                fg = np.array([], dtype=np.int64)

            rest_n = self.batch_size - fg.size
            rest = self.rng.choice(self.all_indices, size=rest_n, replace=True)

            batch = [(int(i), False) for i in rest] + [(int(i), True) for i in fg]
            yield batch

    def __len__(self) -> int:
        n = len(self.all_indices)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
