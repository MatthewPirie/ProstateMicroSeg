# ProstateMicroSeg/src/utils/helper_functions.py

from __future__ import annotations

import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "NO_GIT_REPO"


def _make_run_dir(base_dir: str, run_name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = (base / run_name) if run_name else (base / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_checkpoint(
    ckpt_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    extra: Optional[dict] = None,
) -> None:
    """
    Saves a training checkpoint.
    Stores auxiliary info under payload["extra"] to avoid key collisions.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "extra": extra if extra is not None else {},
    }
    torch.save(payload, ckpt_path)
