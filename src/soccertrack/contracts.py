"""Shared data contracts: schemas, identifiers, validation and table I/O.

Owner: Member 3. Priority milestone M0 (first half-day): every producer validates its output with this module.
Specification: docs/team/SHARED_CONTRACTS.md (v1.0). This module must implement exactly that document.
Changing a constant here requires approval from all four members and a SCHEMA_VERSION bump.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # imported lazily so the skeleton is importable before installation
    import numpy as np
    import pandas as pd

SCHEMA_VERSION = "1.0"

# ---- Table schemas (column order is part of the contract) -------------------------------------------------------
TRACK_COLUMNS = ("frame", "track_id", "x", "y", "w", "h", "score")
DETECTION_COLUMNS = ("frame", "x", "y", "w", "h", "score")
PROPOSAL_COLUMNS = ("frame", "x", "y", "w", "h", "area", "fill_ratio", "score")
PATCH_SCORE_COLUMNS = ("sample_id", "match", "frame", "kind", "y_true", "score")
PITCH_TRACK_COLUMNS = ("frame", "track_id", "x_m", "y_m", "quality")
WORKLOAD_COLUMNS = (
    "match", "track_id", "source", "duration_s", "distance_m", "mean_speed_mps", "peak_speed_mps",
    "high_speed_distance_m", "sprint_distance_m", "n_accel", "n_decel",
)
RAW_MOT_NUM_COLUMNS = 10

TABLE_SCHEMAS = {
    "tracks": TRACK_COLUMNS,
    "detections": DETECTION_COLUMNS,
    "proposals": PROPOSAL_COLUMNS,
    "patch_scores": PATCH_SCORE_COLUMNS,
    "pitch_tracks": PITCH_TRACK_COLUMNS,
    "workload": WORKLOAD_COLUMNS,
}

PATCH_KEYS = ("X", "y", "match", "frame", "box", "kind", "meta_json")
PATCH_SHAPE = (64, 32, 3)  # height, width, RGB channels
PATCH_KINDS = ("gt", "random", "partial", "proposal")

# ---- Identifiers (closed vocabulary) ----------------------------------------------------------------------------
DETECTOR_IDS = (
    "gt_oracle", "gt_noisy",
    "scratch_mwb0", "scratch_hog_svm", "scratch_cnn",
    "yolov8n_coco",
    # Phase 3
    "yolov8n_ft", "yolov8s_ft", "scratch_hog_svm_v2", "scratch_cnn_v2",
)
TRACKER_IDS = ("iou_hungarian", "bytetrack", "iou_hungarian_v2", "bytetrack_tuned")
SPLITS = ("train", "val", "test")
ARTIFACT_KINDS = ("detections", "tracks", "proposals", "patch_scores")
EVAL_TASKS = {
    "E1": "patch_classification",
    "E2": "proposal_classification",
    "E3": "detection",
    "E4": "tracking",
}
REQUIRED_METRIC_KEYS = {
    "patch_classification": ("sensitivity", "specificity", "precision", "f1", "accuracy", "roc_auc", "pr_auc"),
    "proposal_classification": ("sensitivity", "specificity", "precision", "f1", "accuracy", "roc_auc", "pr_auc"),
    "detection": ("precision", "recall", "f1", "ap50", "ap50_95", "fp_per_frame"),
    "tracking": ("hota", "deta", "assa", "loca", "idf1", "mota", "idsw"),
}
REQUIRED_COUNT_KEYS = {
    "patch_classification": ("tp", "fp", "tn", "fn"),
    "proposal_classification": ("tp", "fp", "tn", "fn", "proposals", "gt"),
    "detection": ("frames", "gt", "pred", "tp", "fp", "fn"),
    "tracking": ("frames", "gt_ids", "pred_ids", "gt", "pred"),
}
# Value domains of metric keys (SHARED_CONTRACTS.md Section 1 and Section 8).
BOUNDED_METRIC_KEYS = (  # fractions in [0, 1]
    "precision", "recall", "f1", "ap50", "ap50_95", "sensitivity", "specificity", "accuracy", "roc_auc", "pr_auc",
    "hota", "deta", "assa", "loca", "idf1",
)
MOTA_KEYS = ("mota",)  # <= 1, can be negative
NONNEGATIVE_REAL_KEYS = ("fp_per_frame", "fps")  # >= 0, can exceed 1
INTEGER_METRIC_KEYS = ("idsw",)  # non-negative integer counts (all `counts` values are integers too)

METRICS_REQUIRED_KEYS = (
    "schema_version", "run_id", "eval_id", "task", "method", "split", "match", "frameset",
    "params", "counts", "metrics", "inputs", "provenance",
)
RUN_META_REQUIRED_KEYS = (
    "schema_version", "kind", "method", "tracker", "split", "match", "frameset", "frames",
    "image_width", "image_height", "fps", "score_floor", "operating_threshold", "frozen_test",
    "config_sha256", "git_commit", "created_utc", "runtime_seconds", "frames_per_second", "device", "notes",
)


class ContractError(ValueError):
    """Raised when an artefact violates docs/team/SHARED_CONTRACTS.md."""


def pipeline_id(detector: str, tracker: str) -> str:
    """Return "<detector>+<tracker>" after checking both IDs are registered.

    TODO(Member 3, M0).
    """
    raise NotImplementedError("TODO(Member 3, M0)")


# ---- Validation --------------------------------------------------------------------------------------------------
def validate_tracks(df: "pd.DataFrame", image_size: tuple[int, int] | None = None) -> "pd.DataFrame":
    """Validate a Track table (Section 3); return it unchanged or raise ContractError. image_size = (W, H)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_detections(df: "pd.DataFrame", meta: dict[str, Any] | None = None) -> "pd.DataFrame":
    """Validate a Detection table (Section 4) and, if given, its consistency with the run metadata."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_proposals(df: "pd.DataFrame", meta: dict[str, Any] | None = None) -> "pd.DataFrame":
    """Validate a Proposal table (Section 5)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_patch_set(arrays: dict[str, "np.ndarray"]) -> None:
    """Validate a loaded patch set (Section 6): keys, dtypes, shapes, labels, kinds."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_patch_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate a Patch score table (Section 6.1)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_run_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Validate a run metadata sidecar (Section 7)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Validate a Metrics JSON document (Section 8): required keys, task-specific metric/count keys and the value
    domains (BOUNDED_METRIC_KEYS in [0, 1]; MOTA_KEYS <= 1 and possibly negative; NONNEGATIVE_REAL_KEYS >= 0;
    INTEGER_METRIC_KEYS and counts non-negative integers; null allowed for undefined values)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def validate_pitch_tracks(df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate a PitchTrack table (Section 12, Phase 3)."""
    raise NotImplementedError("TODO(Member 3, M0)")


# ---- I/O (the only sanctioned way to read/write contract tables) ------------------------------------------------
def read_table(path: Path, kind: str) -> "pd.DataFrame":
    """Read a contract CSV of `kind` (key of TABLE_SCHEMAS) with correct dtypes, then validate it."""
    raise NotImplementedError("TODO(Member 3, M0)")


def write_table(df: "pd.DataFrame", path: Path, kind: str) -> Path:
    """Validate, sort and write a contract CSV (boxes 2 decimals, scores 4 decimals, no index)."""
    raise NotImplementedError("TODO(Member 3, M0)")


def read_run_meta(table_path: Path) -> dict[str, Any]:
    """Read and validate the `.meta.json` sidecar that belongs to `table_path`."""
    raise NotImplementedError("TODO(Member 3, M0)")


def write_run_meta(meta: dict[str, Any], table_path: Path) -> Path:
    """Validate and write the `.meta.json` sidecar next to `table_path`."""
    raise NotImplementedError("TODO(Member 3, M0)")
