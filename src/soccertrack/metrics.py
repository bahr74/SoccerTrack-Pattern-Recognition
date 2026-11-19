"""Pure metric functions (no file I/O): IoU, matching, P/R/F1, AP, binary-classification metrics.

Owner: Member 3. `iou_matrix` is a priority milestone (M0) because Member 1 may reuse it.
All functions take NumPy arrays in the contract box convention (x, y, w, h; full-frame pixels).
Hand-computed expectations for the fixtures: SHARED_CONTRACTS.md Section 15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def iou_matrix(boxes_a: "np.ndarray", boxes_b: "np.ndarray") -> "np.ndarray":
    """Pairwise IoU between [N, 4] and [M, 4] xywh boxes -> [N, M]."""
    raise NotImplementedError("TODO(Member 3, M0)")


def match_frame(gt: "np.ndarray", det: "np.ndarray", scores: "np.ndarray", iou_threshold: float) -> dict[str, Any]:
    """Greedy one-to-one matching by descending score for ONE frame -> {"tp_flags", "matched_gt", "fn"}."""
    raise NotImplementedError("TODO(Member 3, M1)")


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1); define 0/0 as 0.0."""
    raise NotImplementedError("TODO(Member 3, M1)")


def average_precision(tp_flags: "np.ndarray", scores: "np.ndarray", num_gt: int) -> tuple[float, dict[str, list]]:
    """All-point interpolated AP over detections pooled across frames; also return the PR curve."""
    raise NotImplementedError("TODO(Member 3, M1)")


def binary_classification_metrics(y_true: "np.ndarray", scores: "np.ndarray", threshold: float) -> dict[str, Any]:
    """Confusion matrix (score >= threshold -> positive), sensitivity, specificity, precision, F1, accuracy,
    ROC-AUC and PR-AUC (sklearn.metrics), plus ROC/PR curves."""
    raise NotImplementedError("TODO(Member 3, M1)")
