"""Phase 3: failure analysis and explainability evaluation.

Owner: Member 3. Reads trained models only through the public interface of classifiers.py (Member 1) and
contract artefacts (Detection/Track tables, Metrics JSON). Failure analysis uses TRAIN/VAL data only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def stratify_detection_errors(detections_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> "pd.DataFrame":
    """Label each GT/detection as TP/FP/FN with height bin, image region and crowding level."""
    raise NotImplementedError("TODO(Member 3, Phase 3)")


def id_switch_events(tracks_csv: Path, gt_tracks: "pd.DataFrame") -> "pd.DataFrame":
    """List frames and GT IDs where the matched predicted ID changes."""
    raise NotImplementedError("TODO(Member 3, Phase 3)")


def gradcam(cnn_classifier: Any, patches_rgb: "np.ndarray") -> "np.ndarray":
    """Grad-CAM heatmaps [N, 64, 32] from cnn_classifier.model / .gradcam_layer / .preprocess."""
    raise NotImplementedError("TODO(Member 3, Phase 3)")


def svm_weight_map(hog_svm_classifier: Any, cfg: dict[str, Any]) -> "np.ndarray":
    """Positive/negative HOG weight visualisation from hog_svm_classifier.hog_weights()."""
    raise NotImplementedError("TODO(Member 3, Phase 3)")


def permutation_importance_groups(classifier: Any, X: "np.ndarray", y: "np.ndarray",
                                  groups: dict[str, Any]) -> "pd.DataFrame":
    """Drop in validation PR-AUC when a feature group (HOG / colour / shape) is permuted."""
    raise NotImplementedError("TODO(Member 3, Phase 3)")
