"""Approach A, stage 1: pitch mask, background model, foreground mask and morphology (label-free).

Owner: Member 1. Config: configs/scratch.yaml -> segmentation, sampling.
Artefacts: outputs/masks/<match>_pitch.png, outputs/backgrounds/<match>.png, outputs/stages/... (Contracts §2, §10).
Fallbacks (plan Section 4.4 and Section 11): keypoint polygon, manual polygon, running median.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def estimate_background(clip_path: Path, num_frames: int, total_frames: int, work_scale: float) -> "np.ndarray":
    """Per-pixel median of `num_frames` frames spread evenly over the clip (BGR uint8, at `work_scale`)."""
    raise NotImplementedError("TODO(Member 1, M3)")


def pitch_mask(background: "np.ndarray", cfg: dict[str, Any], match_id: str,
               calibration_dir: Path | None = None) -> "np.ndarray":
    """Binary pitch mask (bool, same size as `background`) using cfg['segmentation']['pitch_mask_method'].

    hsv: green threshold -> largest component -> fill holes/close.
    keypoints: polygon from <calibration_dir>/<match>_keypoints.json (verify pixel space first).
    manual: polygon from cfg['segmentation']['manual_polygons'][match_id].
    """
    raise NotImplementedError("TODO(Member 1, M3)")


def foreground_mask(frame_small: "np.ndarray", background: "np.ndarray", mask: "np.ndarray",
                    cfg: dict[str, Any]) -> "np.ndarray":
    """|blur(frame) - background| > threshold inside the pitch mask (bool), before morphology."""
    raise NotImplementedError("TODO(Member 1, M3)")


def clean_mask(fg: "np.ndarray", cfg: dict[str, Any]) -> "np.ndarray":
    """Median filter, opening, closing and a small vertical dilation (bool in, bool out)."""
    raise NotImplementedError("TODO(Member 1, M3)")


def save_pitch_mask(mask: "np.ndarray", match_id: str, full_size: tuple[int, int]) -> Path:
    """Write outputs/masks/<match>_pitch.png at FULL resolution, uint8 {0, 255} (Contracts §10)."""
    raise NotImplementedError("TODO(Member 1, M3)")
