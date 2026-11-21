"""Approach A, stage 2: connected components -> candidate boxes (Proposal table, Contracts §5).

Owner: Member 1. Config: configs/scratch.yaml -> proposals.
The perspective size model h_expected(y) is fitted on TRAIN ground truth only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@dataclass
class SizeModel:
    """Expected player box height as a function of the box's bottom row: h = slope * y_bottom + intercept."""

    slope: float
    intercept: float
    width_to_height: float | None = None  # Member 1 provisional extension; fitted on train only

    def expected_height(self, y_bottom: "np.ndarray") -> "np.ndarray":
        """Return the expected height in full-resolution pixels."""
        import numpy as np
        y = np.asarray(y_bottom, dtype=float)
        if not np.isfinite(y).all() or not np.isfinite([self.slope, self.intercept]).all():
            raise ValueError("Size-model coefficients and bottom rows must be finite")
        return self.slope * y + self.intercept


def fit_size_model(train_tracks: "pd.DataFrame") -> SizeModel:
    """Least-squares fit of box height vs. bottom row on TRAIN GT boxes (never val/test)."""
    import numpy as np
    if train_tracks.attrs.get("split", "train") != "train":
        raise PermissionError("Size model must be fitted on train GT only")
    values = train_tracks.loc[:, ["y", "h"]].to_numpy(dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values[:, 1] <= 1).any() or (values[:, 0] < 0).any():
        raise ValueError("Size-model fitting needs at least two finite, valid training boxes")
    bottom = values[:, 0] + values[:, 1]
    if np.unique(bottom).size < 2:
        raise ValueError("Size-model fitting needs distinct bottom rows")
    slope, intercept = np.linalg.lstsq(np.column_stack([bottom, np.ones(len(bottom))]), values[:, 1], rcond=None)[0]
    ratio = None
    if "w" in train_tracks:
        widths = train_tracks.w.to_numpy(dtype=float)
        if not np.isfinite(widths).all() or (widths <= 1).any():
            raise ValueError("Training widths must be finite and > 1")
        ratio = float(np.median(widths / values[:, 1]))
    return SizeModel(float(slope), float(intercept), ratio)


def propose(mask: "np.ndarray", frame_id: int, work_scale: float, size_model: SizeModel,
            cfg: dict[str, Any]) -> "pd.DataFrame":
    """Connected components of a cleaned mask -> Proposal table rows (score=-1.0) in FULL-resolution pixels.

    Filters: min area, height within size_tolerance x h_expected(y); optionally splits blobs wider than
    split_width_factor x expected width.
    """
    import cv2
    import numpy as np
    import pandas as pd
    from soccertrack.contracts import PROPOSAL_COLUMNS
    from soccertrack.segmentation import _mask, _scale

    _mask(mask)
    _scale(work_scale)
    if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)) or frame_id < 0:
        raise ValueError("frame_id must be a nonnegative integer")
    options = cfg["proposals"]
    lo, hi = options["size_tolerance"]
    if not 0 < lo <= hi or options["min_area_px"] < 0 or options["split_width_factor"] <= 1:
        raise ValueError("Invalid proposal size limits")
    # Explicit provisional width rule: learned median train w/h; if the caller
    # supplies a height-only model, use the midpoint of configured h/w bounds.
    ratio = size_model.width_to_height
    if ratio is None:
        aspect = cfg["detector"]["rules"]["aspect_range"]
        ratio = 1 / float(np.mean(aspect))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Expected width/height ratio must be positive")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    rows, agreements = [], []
    for component in range(1, n):
        x, y, w, h, area = stats[component]
        expected = float(size_model.expected_height(np.asarray([(y + h) / work_scale]))[0])
        if expected <= 1 or not lo * expected <= h / work_scale <= hi * expected:
            continue
        pieces = 1
        expected_width = expected * ratio * work_scale
        if options["split_wide_blobs"] and w > options["split_width_factor"] * expected_width:
            pieces = max(2, min(int(np.ceil(w / expected_width)), int(w)))
        edges = np.linspace(x, x + w, pieces + 1, dtype=int)
        for left, right in zip(edges[:-1], edges[1:]):
            occupied = (labels[y:y + h, left:right] == component)
            ys, xs = np.where(occupied)
            if not len(xs):
                continue
            px, py = left + xs.min(), y + ys.min()
            pw, ph = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
            full_area = int(round(len(xs) / work_scale**2))
            if min(pw / work_scale, ph / work_scale) <= 1 or full_area < options["min_area_px"]:
                continue
            rows.append([int(frame_id), px / work_scale, py / work_scale, pw / work_scale,
                         ph / work_scale, full_area, len(xs) / (pw * ph), -1.0])
            actual = ph / work_scale
            agreements.append(float(min(actual / expected, expected / actual)))
    result = pd.DataFrame(rows, columns=PROPOSAL_COLUMNS).astype({
        name: "int64" if name in ("frame", "area") else "float64" for name in PROPOSAL_COLUMNS
    })
    # This in-memory, explicitly provisional context is NOT a shared CSV column.
    result.attrs["member1_provisional_size_agreement"] = agreements
    return result


def proposal_recall(proposals: "pd.DataFrame", gt: "pd.DataFrame", iou_threshold: float) -> float:
    """Fraction of GT boxes covered by at least one proposal at `iou_threshold` (upper bound for Approach A).

    May use metrics.iou_matrix (Member 3) once available.
    """
    import numpy as np
    from soccertrack import metrics
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be in [0,1]")
    if len(gt) == 0:
        raise ValueError("Proposal recall is undefined without ground truth")
    hits = 0
    for frame, truth in gt.groupby("frame"):
        candidates = proposals.loc[proposals.frame == frame, ["x", "y", "w", "h"]].to_numpy(float)
        if len(candidates):
            overlaps = metrics.iou_matrix(truth[["x", "y", "w", "h"]].to_numpy(float), candidates)
            hits += int((np.max(overlaps, axis=1) >= iou_threshold).sum())
    return hits / len(gt)
