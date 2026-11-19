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

    def expected_height(self, y_bottom: "np.ndarray") -> "np.ndarray":
        """Return the expected height in full-resolution pixels."""
        raise NotImplementedError("TODO(Member 1, M3)")


def fit_size_model(train_tracks: "pd.DataFrame") -> SizeModel:
    """Least-squares fit of box height vs. bottom row on TRAIN GT boxes (never val/test)."""
    raise NotImplementedError("TODO(Member 1, M3)")


def propose(mask: "np.ndarray", frame_id: int, work_scale: float, size_model: SizeModel,
            cfg: dict[str, Any]) -> "pd.DataFrame":
    """Connected components of a cleaned mask -> Proposal table rows (score=-1.0) in FULL-resolution pixels.

    Filters: min area, height within size_tolerance x h_expected(y); optionally splits blobs wider than
    split_width_factor x expected width.
    """
    raise NotImplementedError("TODO(Member 1, M3)")


def proposal_recall(proposals: "pd.DataFrame", gt: "pd.DataFrame", iou_threshold: float) -> float:
    """Fraction of GT boxes covered by at least one proposal at `iou_threshold` (upper bound for Approach A).

    May use metrics.iou_matrix (Member 3) once available.
    """
    raise NotImplementedError("TODO(Member 1, M3)")
