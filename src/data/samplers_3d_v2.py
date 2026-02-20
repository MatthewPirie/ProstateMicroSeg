# src/data/samplers_3d_v2.py

from __future__ import annotations

from typing import Iterator, List, Tuple, Union

import numpy as np
from torch.utils.data import Sampler


IndexForce = Tuple[int, bool]


class OversampleForegroundBatchSampler3DV2(Sampler[List[IndexForce]]):
    """
    Batch sampler that yields batches of (dataset_index, force_fg) pairs.

    The dataset is expected to accept either:
      - i (int), or
      - (i, force_fg) where force_fg is bool

    Two modes:

    1) mode="deterministic"
       - For each batch, exactly k = round(batch_size * oversample_foreground_percent)
         samples will have force_fg=True (rest False).
       - With small batch sizes (e.g., 2), this may effectively become 50% even if p=0.33.

    2) mode="probabilistic"
       - For each sample position, force_fg ~ Bernoulli(p).
       - Over many batches, expected forced-FG proportion approaches p.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        oversample_foreground_percent: float = 0.33,
        seed: int = 0,
        drop_last: bool = True,
        mode: str = "probabilistic",  # "probabilistic" | "deterministic"
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 <= oversample_foreground_percent <= 1.0):
            raise ValueError("oversample_foreground_percent must be in [0,1]")

        mode = str(mode).lower()
        if mode not in {"probabilistic", "deterministic"}:
            raise ValueError("mode must be 'probabilistic' or 'deterministic'")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.p = float(oversample_foreground_percent)
        self.drop_last = bool(drop_last)
        self.mode = mode
        self.shuffle = bool(shuffle)

        self.rng = np.random.default_rng(int(seed))

        self.n = len(self.dataset)
        if self.n <= 0:
            raise ValueError("dataset must have positive length")

    def __len__(self) -> int:
        if self.drop_last:
            return self.n // self.batch_size
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[IndexForce]]:
        # Create an ordering of dataset indices for this epoch
        if self.shuffle:
            order = self.rng.permutation(self.n).tolist()
        else:
            order = list(range(self.n))

        batch: List[IndexForce] = []

        if self.mode == "deterministic":
            k = int(np.round(self.batch_size * self.p))
            k = int(np.clip(k, 0, self.batch_size))

        for idx in order:
            # Default is non-forced
            force_fg = False

            if self.mode == "probabilistic":
                force_fg = bool(self.rng.random() < self.p)

            batch.append((int(idx), force_fg))

            if len(batch) == self.batch_size:
                if self.mode == "deterministic" and k > 0:
                    # Flip exactly k positions in this batch to force_fg=True
                    # (chosen uniformly at random among positions)
                    pos = self.rng.choice(self.batch_size, size=k, replace=False)
                    batch = [(bi, (j in set(pos))) for j, (bi, _) in enumerate(batch)]

                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            if self.mode == "deterministic":
                # For the last partial batch, scale k to the actual batch length
                k_last = int(np.round(len(batch) * self.p))
                k_last = int(np.clip(k_last, 0, len(batch)))
                if k_last > 0:
                    pos = self.rng.choice(len(batch), size=k_last, replace=False)
                    pos_set = set(int(x) for x in pos)
                    batch = [(bi, (j in pos_set)) for j, (bi, _) in enumerate(batch)]
            yield batch
