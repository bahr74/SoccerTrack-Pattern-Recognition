"""Approach A end-to-end detector: segmentation -> proposals -> patch classifier -> NMS -> Detection table.

Owner: Member 1. Config: configs/scratch.yaml -> detector (+ segmentation, proposals).
Outputs (Contracts §2, §4, §5, §7):
  outputs/detections/<detector>/<split>/<match>__<frameset>.csv + .meta.json
  outputs/proposals/<detector>/<split>/<match>__<frameset>.csv + .meta.json   (scored, before NMS -> E2)
Detector IDs: scratch_mwb0 (rule verifier), scratch_hog_svm, scratch_cnn.
NMS: torchvision.ops.nms. Write every detection with score >= evaluation.score_floor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def rule_scores(proposals: "pd.DataFrame", cfg: dict[str, Any]) -> "np.ndarray":
    """MWB-0 verifier: score in [0, 1] from aspect ratio, fill ratio and size-model agreement."""
    raise NotImplementedError("TODO(Member 1, M3)")


def run_detection(detector_id: str, match_id: str, split: str, frameset: str, cfg: dict[str, Any],
                  sources: dict[str, Any], frozen_test: bool = False) -> "pd.DataFrame":
    """Run Approach A on every frame of `frameset`; write proposals, detections and metadata; return detections.

    Milestones: M3 for scratch_mwb0 (rule verifier); M7 adds scratch_hog_svm and scratch_cnn.
    """
    raise NotImplementedError("TODO(Member 1, M3/M7)")
