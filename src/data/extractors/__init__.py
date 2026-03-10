# src/data/extractors/__init__.py
"""
Extractor registry.

Goal:
    from src.data.extractors import get_extractor

Then select an extraction strategy by name inside the dataset.
"""

from __future__ import annotations

from typing import Callable, Dict, List

ExtractorFn = Callable[..., tuple]


def _registry() -> Dict[str, ExtractorFn]:
    """
    Map extractor name strings -> extractor functions.

    Imports are kept inside this function so they are lazy and avoid
    circular-import issues.
    """
    from .patch_centered_fg import extract as patch_centered_fg
    from .patch_random_fg import extract as patch_random_fg
    from .full_inplane_zstrided import extract as full_inplane_zstrided
    from .full_inplane_zwindow import extract as full_inplane_zwindow

    return {
        "patch_centered_fg": patch_centered_fg,
        "patch_random_fg": patch_random_fg,
        "full_inplane_zstrided": full_inplane_zstrided,
        "full_inplane_zwindow": full_inplane_zwindow,
    }


def available_extractors() -> List[str]:
    """List valid extractor names."""
    return sorted(_registry().keys())


def get_extractor(name: str) -> ExtractorFn:
    """
    Return the extractor function for a given name.

    Example:
        fn = get_extractor("patch_centered_fg")
        img_out, lbl_out, meta = fn(...)
    """
    reg = _registry()
    if name not in reg:
        raise ValueError(f"Unknown extractor '{name}'. Options: {available_extractors()}")
    return reg[name]