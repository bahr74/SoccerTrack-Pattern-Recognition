"""Phase 3: external-workload indicators from PitchTrack tables.

Owner: Member 4. Config: configs/project.yaml -> workload.
These are EXTERNAL LOAD indicators (distance, speed, acceleration, speed bands). They are not physiological
fatigue or injury measures and must never be reported as such.
Validation: compute the same summary from GT tracks and predicted tracks through the same mapping and report
the per-player difference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def smooth_positions(pitch_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> "pd.DataFrame":
    """Savitzky-Golay smoothing per track (only quality == 1 rows; no interpolation across long gaps)."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def kinematics(pitch_tracks: "pd.DataFrame", fps: float) -> "pd.DataFrame":
    """Add speed (m/s) and acceleration (m/s^2) columns from smoothed positions."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def summarise(kin: "pd.DataFrame", match_id: str, source: str, cfg: dict[str, Any]) -> "pd.DataFrame":
    """Workload summary table (Contracts §12) per track."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")
