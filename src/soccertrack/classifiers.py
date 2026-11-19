"""Patch classifiers for Approach A: HOG + Linear SVM and a Small CNN (both trained from scratch).

Owner: Member 1 (sole owner). Member 3 READS the public interface below for E1 and Phase 3 explainability;
Member 3 never edits this file. Any interface change must be agreed with Member 3 first.

Common interface (Contracts §6): predict_proba(X uint8 [N, 64, 32, 3] RGB) -> float32 [N] in [0, 1].
Normalisation statistics are computed on the TRAIN split only and stored with the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


class HogSvmClassifier:
    """HOG (features.hog_features) + sklearn LinearSVC with balanced class weights.

    Scores: Platt-calibrated probability (CalibratedClassifierCV fitted on TRAIN folds) so outputs are in [0, 1].
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError("TODO(Member 1, M5)")

    def fit(self, X: "np.ndarray", y: "np.ndarray") -> "HogSvmClassifier":
        raise NotImplementedError("TODO(Member 1, M5)")

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        raise NotImplementedError("TODO(Member 1, M5)")

    def hog_weights(self) -> "np.ndarray":
        """Linear SVM weight vector [756] (used by Member 3 for weight maps in Phase 3)."""
        raise NotImplementedError("TODO(Member 1, M5)")

    def save(self, path: Path) -> None:
        raise NotImplementedError("TODO(Member 1, M5)")

    @classmethod
    def load(cls, path: Path) -> "HogSvmClassifier":
        raise NotImplementedError("TODO(Member 1, M5)")


class CnnClassifier:
    """Small CNN: 3 x (Conv3x3-BN-ReLU-MaxPool) with channels [16, 32, 64] -> global average pool -> FC(1).

    About 25k parameters (must stay < 100k). Weighted BCE loss; early stopping on validation PR-AUC.
    Attributes required by Member 3 (Grad-CAM, Phase 3):
      - `model`: the torch.nn.Module
      - `gradcam_layer`: the last convolutional layer module
      - `preprocess(X) -> torch.Tensor`: the exact normalisation used at inference
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        raise NotImplementedError("TODO(Member 1, M6)")

    def fit(self, X_train: "np.ndarray", y_train: "np.ndarray", X_val: "np.ndarray", y_val: "np.ndarray") -> dict:
        """Train; return the history dict {epoch, train_loss, val_loss, val_pr_auc}."""
        raise NotImplementedError("TODO(Member 1, M6)")

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        raise NotImplementedError("TODO(Member 1, M6)")

    def preprocess(self, X: "np.ndarray") -> Any:
        raise NotImplementedError("TODO(Member 1, M6)")

    def save(self, path: Path) -> None:
        raise NotImplementedError("TODO(Member 1, M6)")

    @classmethod
    def load(cls, path: Path) -> "CnnClassifier":
        raise NotImplementedError("TODO(Member 1, M6)")


def load_classifier(detector_id: str) -> HogSvmClassifier | CnnClassifier:
    """Load the trained model of `scratch_hog_svm` / `scratch_cnn` from outputs/models/<detector_id>/."""
    raise NotImplementedError("TODO(Member 1, M6)")
