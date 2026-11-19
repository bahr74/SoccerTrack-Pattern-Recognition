"""Visualisation: overlays, EDA figures, segmentation stage figures and Phase 3 trajectory/heatmap figures.

Owner: Member 4. Consumes contract artefacts only (Track/Detection tables, EDA stats JSON, stage images,
PitchTrack tables). Evaluation figures (confusion matrices, PR/ROC curves) belong to compare.py (Member 3).
Writes to outputs/figures/{eda,overlays,workload}/ ; curated copies go to docs/phase2|3/figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def draw_boxes(frame_bgr: "np.ndarray", table: "pd.DataFrame", frame_id: int, color_by: str = "track_id",
               label: bool = True) -> "np.ndarray":
    """Draw boxes of one frame from a Track or Detection table; stable colour per track_id."""
    raise NotImplementedError("TODO(Member 4, M1)")


def side_by_side(frame_bgr: "np.ndarray", left: "pd.DataFrame", right: "pd.DataFrame", frame_id: int,
                 titles: tuple[str, str]) -> "np.ndarray":
    """Two overlays (e.g. Pipeline A vs Pipeline B) stacked vertically for wide panoramas."""
    raise NotImplementedError("TODO(Member 4, M2)")


def overlay_video(clip_path: Path, tables: dict[str, "pd.DataFrame"], frames: range, out_path: Path,
                  scale: float = 0.5) -> Path:
    """Write an MP4 overlay (git-ignored) for the given frames; frames decoded with data_io.iter_frames."""
    raise NotImplementedError("TODO(Member 4, M2)")


def plot_eda(stats: dict[str, Any], box_sample: "pd.DataFrame", out_dir: Path) -> list[Path]:
    """Boxes per match, box width/height histograms, height vs. image row, track lengths, issues table."""
    raise NotImplementedError("TODO(Member 4, M1)")


def plot_stage_figure(stage_dir: Path, frame_id: int, out_path: Path) -> Path:
    """Compose frame / foreground / morphology / proposals images into one report figure."""
    raise NotImplementedError("TODO(Member 4, M2)")


def plot_trajectories(pitch_tracks: "pd.DataFrame", out_path: Path) -> Path:
    """Phase 3: player trajectories on a 105 x 68 m pitch drawing."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")


def plot_heatmap(pitch_tracks: "pd.DataFrame", track_id: int, out_path: Path, bins: tuple[int, int] = (21, 14)) -> Path:
    """Phase 3: occupancy heatmap of one player on the pitch."""
    raise NotImplementedError("TODO(Member 4, Phase 3)")
