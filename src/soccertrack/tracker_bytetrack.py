"""Pipeline B tracker: adapter around Ultralytics BYTETracker (tracker ID `bytetrack`).

Owner: Member 2. Config: configs/yolo.yaml -> bytetrack.
Independent of the from-scratch tracker (Member 1): never import tracker_iou.py.
Input: Detection table (+ meta) of a stride-1 frameset. Output: Track table (+ meta), Contracts §3, §7.

Verified interface (ultralytics 8.4.182, ultralytics/trackers/byte_tracker.py):
  BYTETracker(args) where args has track_high_thresh, track_low_thresh, new_track_thresh, track_buffer,
  match_thresh, fuse_score.
  update(results, img=None) where `results` exposes NumPy arrays `conf`, `cls`, `xywh` (centre x, centre y, w, h)
  and supports boolean indexing; returns rows [x1, y1, x2, y2, track_id, score, cls, idx].
  `idx` is the index of the associated input detection -> lets us write the raw detection box
  (yolo.bytetrack.output_box = detection) instead of the Kalman box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class DetectionsForByteTrack:
    """Minimal results-like object built from one frame of a Detection table (conf, cls, xywh, __getitem__, __len__)."""

    def __init__(self, frame_dets: "pd.DataFrame") -> None:
        raise NotImplementedError("TODO(Member 2, M4)")


def track_sequence(detections: "pd.DataFrame", frames: range, cfg: dict[str, Any]) -> "pd.DataFrame":
    """Feed every frame of `frames` (stride 1, empty frames included) to BYTETracker; return the Track table."""
    raise NotImplementedError("TODO(Member 2, M4)")
