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
    """Greedy one-to-one matching by descending score for ONE frame -> {"tp_flags", "matched_gt", "fn"}.

    The most confident detection picks first: it takes the still-unmatched GT box with the highest IoU, if that IoU
    is >= iou_threshold (TP); otherwise it is a false positive. Each GT box can be taken only once, so a second box
    on the same player is a FP (duplicate). GT boxes nobody took are false negatives.

    Returns arrays aligned with the INPUT order of `det`:
      tp_flags   bool [M]  True = true positive
      matched_gt int  [M]  index of the GT box it matched, -1 if none
      matched_iou float [M] IoU with that GT box (0 if none)
      fn         int       number of unmatched GT boxes
      gt_matched bool [N]  True for GT boxes that were found
    """
    import numpy as np

    gt = np.asarray(gt, dtype=float).reshape(-1, 4)
    det = np.asarray(det, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    ious = iou_matrix(det, gt)  # [M, N]
    tp_flags = np.zeros(len(det), dtype=bool)
    matched_gt = np.full(len(det), -1, dtype=int)
    matched_iou = np.zeros(len(det))
    gt_taken = np.zeros(len(gt), dtype=bool)
    for d in np.argsort(-scores, kind="stable"):  # highest score first; ties keep input order
        if len(gt) == 0:
            break
        candidate = np.where(gt_taken, -1.0, ious[d])  # GT boxes already taken cannot be matched again
        g = int(np.argmax(candidate))
        if candidate[g] >= iou_threshold:
            tp_flags[d], matched_gt[d], matched_iou[d] = True, g, candidate[g]
            gt_taken[g] = True
    return {"tp_flags": tp_flags, "matched_gt": matched_gt, "matched_iou": matched_iou,
            "fn": int((~gt_taken).sum()), "gt_matched": gt_taken}


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1); define 0/0 as 0.0.

    precision = TP / (TP + FP)   "of what I predicted, how much was right?"
    recall    = TP / (TP + FN)   "of what exists, how much did I find?"
    F1        = 2 P R / (P + R)  harmonic mean, low if either one is low
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def _downsample(n: int, max_points: int) -> "np.ndarray":
    """Indices of at most max_points evenly spaced points, always keeping the first and the last."""
    import numpy as np

    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def average_precision(tp_flags: "np.ndarray", scores: "np.ndarray", num_gt: int,
                      max_points: int = 200) -> tuple[float | None, dict[str, list]]:
    """All-point interpolated AP over detections pooled across frames; also return the PR curve.

    1. Sort all detections (from every frame) by score, highest first.
    2. Walk down the list; after each detection compute precision and recall so far.
    3. Make precision non-increasing from right to left (the "envelope": at each recall, the best precision
       reachable at that recall or higher).
    4. AP = area under that step curve = sum over each recall step of (step width x envelope precision).
    Returns (None, empty curve) when there is no GT box (AP undefined).
    """
    import numpy as np

    if num_gt <= 0:
        return None, {"recall": [], "precision": []}
    tp_flags = np.asarray(tp_flags, dtype=bool).reshape(-1)
    order = np.argsort(-np.asarray(scores, dtype=float).reshape(-1), kind="stable")
    tp = tp_flags[order].astype(float)
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # Sentinels: recall starts at 0; the envelope runs from right to left.
    r = np.concatenate([[0.0], recall])
    p = np.concatenate([[1.0 if len(tp) else 0.0], precision])
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    ap = float(np.sum((r[1:] - r[:-1]) * envelope[1:]))
    keep = _downsample(len(recall), max_points)
    curve = {"recall": [round(float(v), 4) for v in recall[keep]],
             "precision": [round(float(v), 4) for v in precision[keep]]}
    return ap, curve


def binary_classification_metrics(y_true: "np.ndarray", scores: "np.ndarray", threshold: float,
                                  max_points: int = 200) -> dict[str, Any]:
    """Confusion matrix (score >= threshold -> positive), sensitivity, specificity, precision, F1, accuracy,
    ROC-AUC and PR-AUC (sklearn.metrics), plus ROC/PR curves.

    sensitivity = TP / (TP + FN)  (= recall: players correctly called "player")
    specificity = TN / (TN + FP)  (background correctly called "background")
    ROC-AUC = probability that a random positive gets a higher score than a random negative.
    PR-AUC  = average precision of the scores; better than ROC-AUC when negatives dominate (E2).
    ROC-AUC / PR-AUC are None (null) when only one class is present.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    y = np.asarray(y_true).astype(int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    precision, sensitivity, f1 = precision_recall_f1(tp, fp, fn)
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    both_classes = len(np.unique(y)) == 2
    curves: dict[str, dict[str, list]] = {}
    roc_auc = pr_auc = None
    if both_classes:
        roc_auc = float(roc_auc_score(y, s))
        pr_auc = float(average_precision_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        k = _downsample(len(fpr), max_points)
        curves["roc"] = {"fpr": [round(float(v), 4) for v in fpr[k]], "tpr": [round(float(v), 4) for v in tpr[k]]}
        prec, rec, _ = precision_recall_curve(y, s)
        k = _downsample(len(prec), max_points)
        curves["pr"] = {"recall": [round(float(v), 4) for v in rec[k]],
                        "precision": [round(float(v), 4) for v in prec[k]]}
    return {
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "metrics": {"sensitivity": float(sensitivity), "specificity": float(specificity),
                    "precision": float(precision), "f1": float(f1), "accuracy": float(accuracy),
                    "roc_auc": roc_auc, "pr_auc": pr_auc},
        "curves": curves,
    }
