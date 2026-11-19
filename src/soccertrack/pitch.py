"""Phase 3: image -> pitch mapping (metres) from SoccerTrack calibration files.

Owner: Member 4. Input: Track table + <calibration_dir>/<match>_keypoints.json and _camera_intrinsics.npz
(read-only). For 132831 use the corrected keypoints from the official repository (data_corrections/).
Output: PitchTrack table (Contracts §12). Foot point = bottom-centre of the box.
Report the leave-one-out landmark error in metres (reference: median ~2.6 m on 117093 in earlier work).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def load_calibration(match_id: str, calibration_dir: Path) -> dict[str, Any]:
    """Load keypoints (image <-> pitch correspondences) and camera intrinsics; verify the axis convention."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def fit_mapping(calibration: dict[str, Any]) -> Any:
    """Undistort keypoints with the intrinsics and fit a homography (cv2.findHomography, RANSAC)."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def leave_one_out_error(calibration: dict[str, Any]) -> "np.ndarray":
    """Per-landmark reprojection error in metres when each keypoint is held out."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def to_pitch(tracks: "pd.DataFrame", mapping: Any, pitch_size_m: tuple[float, float]) -> "pd.DataFrame":
    """Map foot points to metres; quality = 1 inside the pitch, else 0."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")
