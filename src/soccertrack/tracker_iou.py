"""Pipeline A tracker: from-scratch IoU + Hungarian multi-object tracker (tracker ID `iou_hungarian`).

Owner: Member 1. Config: configs/scratch.yaml -> tracker_iou.
Independent of ByteTrack (Member 2): never import tracker_bytetrack.py.
Input: Detection table (+ meta) of a stride-1 frameset. Output: Track table (+ meta), Contracts §3, §7.
Validate first on gt_oracle and gt_noisy detections, then on scratch detections.
Algorithm: cost = 1 - IoU; scipy.optimize.linear_sum_assignment; reject pairs with IoU < iou_gate;
tentative tracks confirmed after n_init hits; deleted after max_age missed frames; only confirmed tracks are output.
Phase 3 (iou_hungarian_v2): constant-velocity Kalman prediction and two-stage (high/low score) association.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class IouHungarianTracker:
    """Online tracker; call `update` once per frame in increasing frame order."""

    def __init__(self, iou_gate: float, n_init: int, max_age: int, min_score: float) -> None:
        raise NotImplementedError("TODO(Member 1, M8)")

    def update(self, frame_id: int, detections: "pd.DataFrame") -> "pd.DataFrame":
        """Associate this frame's detections; return Track-table rows for confirmed tracks in this frame."""
        raise NotImplementedError("TODO(Member 1, M8)")


def track_sequence(detections: "pd.DataFrame", frames: range, cfg: dict[str, Any]) -> "pd.DataFrame":
    """Run the tracker over `frames` (stride 1; frames without detections still advance the tracker)."""
    raise NotImplementedError("TODO(Member 1, M8)")
