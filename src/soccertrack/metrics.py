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
    """Pairwise IoU between [N, 4] and [M, 4] xywh boxes -> [N, M].

    IoU = area(A ∩ B) / area(A ∪ B) = inter / (area_a + area_b - inter). 1 = identical boxes, 0 = no overlap.
    Vectorised with broadcasting: row i, column j compares boxes_a[i] with boxes_b[j].
    """
    import numpy as np

    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    # Corners: (x1, y1) = top-left, (x2, y2) = bottom-right.
    ax1, ay1, ax2, ay2 = a[:, 0, None], a[:, 1, None], (a[:, 0] + a[:, 2])[:, None], (a[:, 1] + a[:, 3])[:, None]
    bx1, by1, bx2, by2 = b[None, :, 0], b[None, :, 1], (b[:, 0] + b[:, 2])[None, :], (b[:, 1] + b[:, 3])[None, :]
    # Width/height of the overlap rectangle; negative means "no overlap", so clip at 0.
    inter_w = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    inter_h = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = inter_w * inter_h
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


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
