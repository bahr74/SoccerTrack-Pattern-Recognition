"""Evaluation drivers E1-E4, oracle detection inputs and the TrackEval adapter.

Owner: Member 3. Config: configs/evaluation.yaml.
Reads only contract artefacts (Detection/Track/Proposal/Patch score tables + metadata) and writes Metrics JSON
(Contracts §8) to outputs/metrics/<run_id>.json. Never imports detector or tracker modules.
The SAME functions evaluate Pipeline A and Pipeline B (fair comparison).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# ---- Oracle inputs for tracker validation (used by Members 1 and 2) --------------------------------------------
def gt_as_detections(gt_tracks: "pd.DataFrame") -> "pd.DataFrame":
    """Detector `gt_oracle`: drop track_id, keep score 1.0."""
    raise NotImplementedError("TODO(Member 3, M2)")


def noisy_gt_detections(gt_tracks: "pd.DataFrame", drop_rate: float, jitter_px: float, seed: int) -> "pd.DataFrame":
    """Detector `gt_noisy`: randomly drop boxes, jitter coordinates, assign scores in [0.3, 1.0]; clip to image."""
    raise NotImplementedError("TODO(Member 3, M2)")


# ---- E1-E4 -------------------------------------------------------------------------------------------------------
def evaluate_patches(scores_csv: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """E1 from a Patch score table + metadata (operating_threshold from the metadata)."""
    raise NotImplementedError("TODO(Member 3, M1)")


def evaluate_proposals(proposals_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> dict[str, Any]:
    """E2: label scored proposals by IoU >= proposal_eval.iou_positive with GT; classification metrics."""
    raise NotImplementedError("TODO(Member 3, M1)")


def evaluate_detections(detections_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> dict[str, Any]:
    """E3: restrict GT to the metadata frameset; P/R/F1 at the operating threshold for each IoU threshold;
    AP@0.5 and AP@[.5:.95] over all detections; recall by height bin; FP per frame; FPS from metadata."""
    raise NotImplementedError("TODO(Member 3, M1)")


def export_trackeval(gt_tracks: "pd.DataFrame", tracks_csv: Path, run_id: str) -> Path:
    """Write the TrackEval workspace (Contracts §9): 1-based frames, GT conf/class/vis = 1, seqinfo.ini.

    Target output for the fixtures: tests/fixtures/trackeval_example/.
    """
    raise NotImplementedError("TODO(Member 3, M3)")


def evaluate_tracking(tracks_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> dict[str, Any]:
    """E4: export, run TrackEval (HOTA, CLEAR, Identity) with SKIP_SPLIT_FOL=True, return Metrics JSON content."""
    raise NotImplementedError("TODO(Member 3, M3)")


def pool_metrics(per_match: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled result (match = "pooled"): recompute from summed counts where possible, never average ratios."""
    raise NotImplementedError("TODO(Member 3, M4)")


def write_metrics(metrics: dict[str, Any]) -> Path:
    """Validate with contracts.validate_metrics and write outputs/metrics/<run_id>.json."""
    raise NotImplementedError("TODO(Member 3, M1)")
