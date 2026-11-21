"""Approach A, stage 1: pitch mask, background model, foreground mask and morphology (label-free).

Owner: Member 1. Config: configs/scratch.yaml -> segmentation, sampling.
Artefacts: outputs/masks/<match>_pitch.png, outputs/backgrounds/<match>.png, outputs/stages/... (Contracts §2, §10).
Fallbacks (plan Section 4.4 and Section 11): keypoint polygon, manual polygon, running median.
"""

from __future__ import annotations

from pathlib import Path
from numbers import Integral
from typing import TYPE_CHECKING, Any

from soccertrack import config, data_io

if TYPE_CHECKING:
    import numpy as np


def estimate_background(clip_path: Path, num_frames: int, total_frames: int, work_scale: float) -> "np.ndarray":
    """Per-pixel median of `num_frames` frames spread evenly over the clip (BGR uint8, at `work_scale`)."""
    import cv2
    import numpy as np

    _positive_integer(num_frames, "num_frames")
    _positive_integer(total_frames, "total_frames")
    _scale(work_scale)
    ids = np.unique(np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=np.int64))
    sampled = []
    shape = None
    for _, image in data_io.iter_frames(clip_path, ids):
        _image(image)
        if shape is None:
            shape = image.shape
        elif image.shape != shape:
            raise ValueError("Video resolution changed while estimating background")
        size = (int(image.shape[1] * work_scale), int(image.shape[0] * work_scale))
        if min(size) < 1:
            raise ValueError("work_scale makes the image empty")
        sampled.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    if len(sampled) != len(ids):
        raise ValueError("Video ended before all background sample frames were decoded")
    return np.median(np.stack(sampled), axis=0).astype(np.uint8)


def pitch_mask(background: "np.ndarray", cfg: dict[str, Any], match_id: str,
               calibration_dir: Path | None = None) -> "np.ndarray":
    """Binary pitch mask (bool, same size as `background`) using cfg['segmentation']['pitch_mask_method'].

    hsv: green threshold -> largest component -> fill holes/close.
    keypoints: polygon from <calibration_dir>/<match>_keypoints.json (verify pixel space first).
    manual: polygon from cfg['segmentation']['manual_polygons'][match_id].
    """
    import cv2
    import numpy as np
    from scipy.ndimage import binary_fill_holes

    _image(background)
    settings = cfg["segmentation"]
    method = settings["pitch_mask_method"]
    if method == "keypoints":
        raise NotImplementedError(
            "Keypoint fallback requires a verified calibration JSON schema and clip pixel space; use hsv or manual"
        )
    if method == "manual":
        scale = settings["work_scale"]
        _scale(scale)
        points = np.asarray(settings["manual_polygons"][match_id], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Manual pitch polygon must contain at least three finite full-frame xy points")
        points = points * scale
        # Clip before int32 conversion, preventing overflow for malformed huge coordinates.
        points[:, 0] = points[:, 0].clip(0, background.shape[1] - 1)
        points[:, 1] = points[:, 1].clip(0, background.shape[0] - 1)
        points = np.rint(points).astype(np.int32)
        if abs(cv2.contourArea(points)) < 1:
            raise ValueError("Manual pitch polygon has zero visible area")
        mask = np.zeros(background.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)
    if method != "hsv":
        raise ValueError(f"Unknown pitch_mask_method: {method!r}")
    low, high = np.asarray(settings["hsv_green_low"]), np.asarray(settings["hsv_green_high"])
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all():
        raise ValueError("HSV bounds must be finite triples")
    if (low < 0).any() or (high > [179, 255, 255]).any() or (low > high).any():
        raise ValueError("HSV bounds must be ordered and within OpenCV HSV limits")
    green = cv2.inRange(cv2.cvtColor(background, cv2.COLOR_BGR2HSV), low.astype(np.uint8), high.astype(np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(green, connectivity=8)
    if count == 1:
        return np.zeros(background.shape[:2], dtype=bool)
    largest = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    filled = binary_fill_holes(largest).astype(np.uint8)
    kernel = _kernel(settings["close_ksize"], "close_ksize")
    return cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel).astype(bool)


def foreground_mask(frame_small: "np.ndarray", background: "np.ndarray", mask: "np.ndarray",
                    cfg: dict[str, Any]) -> "np.ndarray":
    """|blur(frame) - background| > threshold inside the pitch mask (bool), before morphology."""
    import cv2
    import numpy as np

    _image(frame_small)
    _image(background)
    _mask(mask)
    if frame_small.shape != background.shape or mask.shape != frame_small.shape[:2]:
        raise ValueError("Frame, background and pitch mask must have matching resolutions")
    settings = cfg["segmentation"]
    k = settings["blur_ksize"]
    _kernel(k, "blur_ksize")
    if k % 2 != 1:
        raise ValueError("blur_ksize must be odd")
    colors = {"lab": cv2.COLOR_BGR2LAB, "gray": cv2.COLOR_BGR2GRAY}
    if settings["diff_colorspace"] not in colors:
        raise ValueError("diff_colorspace must be lab or gray")
    threshold = settings["diff_threshold"]
    if not np.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError("diff_threshold must be in [0, 255]")
    code = colors[settings["diff_colorspace"]]
    current = cv2.cvtColor(cv2.GaussianBlur(frame_small, (k, k), 0), code)
    reference = cv2.cvtColor(cv2.GaussianBlur(background, (k, k), 0), code)
    difference = cv2.absdiff(current, reference)
    if difference.ndim == 3:
        difference = difference.max(axis=2)
    return (difference > threshold) & mask


def clean_mask(fg: "np.ndarray", cfg: dict[str, Any]) -> "np.ndarray":
    """Median filter, opening, closing and a small vertical dilation (bool in, bool out)."""
    import cv2
    import numpy as np

    _mask(fg)
    settings = cfg["segmentation"]
    median = settings["median_ksize"]
    _positive_integer(median, "median_ksize")
    if median % 2 != 1:
        raise ValueError("median_ksize must be odd")
    image = fg.astype(np.uint8) * 255
    if median > 1:
        image = cv2.medianBlur(image, median)
    for operation, key in ((cv2.MORPH_OPEN, "open_ksize"), (cv2.MORPH_CLOSE, "close_ksize")):
        image = cv2.morphologyEx(image, operation, _kernel(settings[key], key))
    vertical = settings["vertical_dilate"]
    _positive_integer(vertical, "vertical_dilate")
    return cv2.dilate(image, np.ones((vertical, 1), dtype=np.uint8)).astype(bool)


def save_pitch_mask(mask: "np.ndarray", match_id: str, full_size: tuple[int, int]) -> Path:
    """Write outputs/masks/<match>_pitch.png at FULL resolution, uint8 {0, 255} (Contracts §10)."""
    import cv2
    import numpy as np

    _mask(mask)
    if not isinstance(match_id, str) or not match_id.isascii() or not match_id.isdigit():
        raise ValueError("match_id must be a string of ASCII digits")
    if len(full_size) != 2:
        raise ValueError("full_size must be (width, height)")
    for value in full_size:
        _positive_integer(value, "full_size")
    root = config.repo_root().resolve()
    settings = config.load_config()["project"]["paths"]
    outputs = (root / settings["outputs_root"]).resolve()
    path = (outputs / "masks" / f"{match_id}_pitch.png").resolve()
    if outputs == root or not outputs.is_relative_to(root) or not path.is_relative_to(outputs):
        raise ValueError("Pitch mask output must stay inside repository outputs")
    image = cv2.resize(mask.astype(np.uint8), tuple(full_size), interpolation=cv2.INTER_NEAREST) * 255
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write pitch mask: {path}")
    return path


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _scale(value: float) -> None:
    import numpy as np
    if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("work_scale must be in (0, 1]")


def _image(value: "np.ndarray") -> None:
    import numpy as np
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) == 0:
        raise ValueError("Expected a nonempty BGR uint8 image of shape (H, W, 3)")


def _mask(value: "np.ndarray") -> None:
    import numpy as np
    if not isinstance(value, np.ndarray) or value.dtype != bool or value.ndim != 2 or min(value.shape) == 0:
        raise ValueError("Expected a nonempty boolean mask of shape (H, W)")


def _kernel(size: int, name: str) -> "np.ndarray":
    import numpy as np
    _positive_integer(size, name)
    return np.ones((size, size), dtype=np.uint8)
