"""Patch dataset builder for the Approach A classifiers (Patch set contract, Contracts §6).

Owner: Member 1. Config: configs/scratch.yaml -> patches, sampling.
Test-split patches may only be built inside a frozen test run (split.require_test_access).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def crop_patch(frame_bgr: "np.ndarray", box: tuple[float, float, float, float], context: float = 0.10,
               size: tuple[int, int] = (64, 32)) -> "np.ndarray":
    """Crop box (+context margin, padded at borders), resize to (height, width) and return RGB uint8."""
    raise NotImplementedError("TODO(Member 1, M4)")


def build_patch_set(split: str, cfg: dict[str, Any], sources: dict[str, Any],
                    frozen_test: bool = False) -> dict[str, "np.ndarray"]:
    """Sample positives (GT) and negatives (random / partial / proposal) at a 1:3 ratio; return the arrays."""
    raise NotImplementedError("TODO(Member 1, M4)")


def save_patch_set(arrays: dict[str, "np.ndarray"], path: Path) -> Path:
    """Validate with contracts.validate_patch_set and write with numpy.savez_compressed."""
    raise NotImplementedError("TODO(Member 1, M4)")


def load_patch_set(path: Path) -> dict[str, "np.ndarray"]:
    """Load and validate a patch set."""
    raise NotImplementedError("TODO(Member 1, M4)")
