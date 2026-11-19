"""Hand-crafted features for 64x32 patches: HOG (implemented from scratch in NumPy) and colour histograms.

Owner: Member 1. Config: configs/scratch.yaml -> hog.
HOG for a 64x32 patch with 8x8 cells, 2x2-cell blocks (stride 1 cell), 9 unsigned bins, L2-Hys:
7 x 3 blocks x 36 values = 756 features. Validate against cv2.HOGDescriptor in tests (tolerance documented).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

HOG_DIM_64x32 = 756


def hog_features(patches_rgb: "np.ndarray", cfg: dict[str, Any]) -> "np.ndarray":
    """HOG descriptors for uint8 RGB patches [N, 64, 32, 3] -> float32 [N, 756] (grayscale, L2-Hys)."""
    raise NotImplementedError("TODO(Member 1, M5)")


def hog_glyph(descriptor: "np.ndarray", cfg: dict[str, Any]) -> "np.ndarray":
    """Render one HOG descriptor as an orientation-glyph image (for the report and Phase 3 explainability)."""
    raise NotImplementedError("TODO(Member 1, M5)")


def color_histograms(patches_rgb: "np.ndarray", bins: tuple[int, int, int]) -> "np.ndarray":
    """Normalised HSV histograms [N, prod(bins)] (optional extra features)."""
    raise NotImplementedError("TODO(Member 1, M5)")
