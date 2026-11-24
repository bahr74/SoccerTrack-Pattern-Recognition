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
BOX_SAMPLE_COLUMNS = ("match", "split", "frame", "x", "y", "w", "h")  # Section 11: outputs/eda/box_sample.csv
RAW_MOT_NUM_COLUMNS = 10

TABLE_SCHEMAS = {
    "tracks": TRACK_COLUMNS,
    "detections": DETECTION_COLUMNS,
    "proposals": PROPOSAL_COLUMNS,
    "patch_scores": PATCH_SCORE_COLUMNS,
    "pitch_tracks": PITCH_TRACK_COLUMNS,
    "workload": WORKLOAD_COLUMNS,
    "box_sample": BOX_SAMPLE_COLUMNS,
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


# ---- Small internal helpers --------------------------------------------------------------------------------------
# Box columns are written with 2 decimals; everything that is a probability-like value with 4 decimals.
_BOX_COLUMNS = ("x", "y", "w", "h")
_INT_COLUMNS = {"frame", "track_id", "sample_id", "y_true", "area", "quality", "n_accel", "n_decel"}
_STR_COLUMNS = {"match", "kind", "split", "source"}
_DECIMALS = {"x": 2, "y": 2, "w": 2, "h": 2, "x_m": 2, "y_m": 2, "score": 4, "fill_ratio": 4}
_DEFAULT_DECIMALS = 4
# Rows are always written in this order (Sections 3 and 4).
_SORT_KEYS = {
    "tracks": (["frame", "track_id"], [True, True]),
    "detections": (["frame", "score"], [True, False]),
    "proposals": (["frame", "score"], [True, False]),
    "patch_scores": (["sample_id"], [True]),
    "pitch_tracks": (["frame", "track_id"], [True, True]),
    "workload": (["match", "track_id", "source"], [True, True, True]),
    "box_sample": (["match", "frame"], [True, True]),
}
# Values are written with 2 or 4 decimals, so a box may exceed the image by up to half a rounding step.
_EPS = 0.006
_ISO_UTC = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"


def _fail(what: str, message: str) -> None:
    raise ContractError(f"{what}: {message}")


def _check_columns(df: "pd.DataFrame", kind: str) -> None:
    """Exact column names in the exact order (the order is part of the contract)."""
    expected = TABLE_SCHEMAS[kind]
    if tuple(df.columns) != tuple(expected):
        _fail(kind, f"columns must be {list(expected)} in this order, got {list(df.columns)}")


def _check_no_nan(df: "pd.DataFrame", kind: str) -> None:
    if df.isna().to_numpy().any():
        bad = [c for c in df.columns if df[c].isna().any()]
        _fail(kind, f"NaN/empty values in column(s) {bad}")


def _numeric(df: "pd.DataFrame", columns, kind: str) -> "pd.DataFrame":
    """Return the given columns as floats, failing with a clear message if one is not numeric."""
    import pandas as pd

    out = {}
    for c in columns:
        values = pd.to_numeric(df[c], errors="coerce")
        if values.isna().any():
            _fail(kind, f"column '{c}' must be numeric")
        out[c] = values.astype(float)
    return pd.DataFrame(out, index=df.index)


def _check_integer_column(df: "pd.DataFrame", column: str, kind: str, minimum: int | None = None) -> None:
    import numpy as np

    values = _numeric(df, [column], kind)[column].to_numpy()
    if not np.all(np.equal(np.mod(values, 1), 0)):
        _fail(kind, f"column '{column}' must contain integers")
    if minimum is not None and len(values) and values.min() < minimum:
        _fail(kind, f"column '{column}' must be >= {minimum} (found {values.min():g})")


def _check_boxes(df: "pd.DataFrame", kind: str, image_size: tuple[int, int] | None) -> None:
    """Box validity of Section 1: w > 1, h > 1, x >= 0, y >= 0 and, if the image size is known, inside the image."""
    b = _numeric(df, _BOX_COLUMNS, kind)
    if (b.w <= 1).any() or (b.h <= 1).any():
        _fail(kind, "box width and height must be > 1 pixel (negative or degenerate box found)")
    if (b.x < -_EPS).any() or (b.y < -_EPS).any():
        _fail(kind, "box x and y must be >= 0 (boxes are clipped to the image before writing)")
    if image_size is not None:
        width, height = image_size
        if (b.x + b.w > width + _EPS).any() or (b.y + b.h > height + _EPS).any():
            _fail(kind, f"box extends outside the {width}x{height} image")


def _check_unit_interval(values, kind: str, column: str) -> None:
    if len(values) and ((values < 0).any() or (values > 1).any()):
        _fail(kind, f"column '{column}' must lie in [0, 1]")


def _frames_from_meta(meta: dict[str, Any]) -> range:
    f = meta["frames"]
    return range(int(f["start"]), int(f["stop"]), int(f["stride"]))


def _check_frames_in_frameset(df: "pd.DataFrame", meta: dict[str, Any], kind: str) -> None:
    """Every row must belong to a frame the producer declared as processed (Section 7: frames)."""
    import numpy as np

    frames = _frames_from_meta(meta)
    f = df["frame"].to_numpy(dtype=np.int64)
    inside = (f >= frames.start) & (f < frames.stop) & ((f - frames.start) % frames.step == 0)
    if not inside.all():
        bad = sorted(set(f[~inside].tolist()))[:5]
        _fail(kind, f"frame(s) {bad} are outside the metadata frameset {meta.get('frameset')} = {frames}")


def _image_size_from_meta(meta: dict[str, Any] | None) -> tuple[int, int] | None:
    if meta and meta.get("image_width") and meta.get("image_height"):
        return int(meta["image_width"]), int(meta["image_height"])
    return None


def pipeline_id(detector: str, tracker: str) -> str:
    """Return "<detector>+<tracker>" after checking both IDs are registered (Section 1.1, closed vocabulary)."""
    if detector not in DETECTOR_IDS:
        raise ContractError(f"unknown detector ID '{detector}'; registered: {list(DETECTOR_IDS)}")
    if tracker not in TRACKER_IDS:
        raise ContractError(f"unknown tracker ID '{tracker}'; registered: {list(TRACKER_IDS)}")
    return f"{detector}+{tracker}"


def split_pipeline_id(pipeline: str) -> tuple[str, str]:
    """Inverse of pipeline_id: "<detector>+<tracker>" -> (detector, tracker), both checked."""
    if pipeline.count("+") != 1:
        raise ContractError(f"pipeline ID must look like '<detector>+<tracker>', got '{pipeline}'")
    detector, tracker = pipeline.split("+")
    pipeline_id(detector, tracker)
    return detector, tracker


# ---- Validation --------------------------------------------------------------------------------------------------
def validate_tracks(df: "pd.DataFrame", image_size: tuple[int, int] | None = None) -> "pd.DataFrame":
    """Validate a Track table (Section 3); return it unchanged or raise ContractError. image_size = (W, H)."""
    kind = "tracks"
    _check_columns(df, kind)
    if df.empty:
        return df
    _check_no_nan(df, kind)
    _check_integer_column(df, "frame", kind, minimum=0)
    _check_integer_column(df, "track_id", kind, minimum=1)
    _check_boxes(df, kind, image_size)
    _check_unit_interval(_numeric(df, ["score"], kind)["score"], kind, "score")
    if df.duplicated(["frame", "track_id"]).any():
        _fail(kind, "duplicate (frame, track_id) pairs")
    keys = list(zip(df["frame"].astype(int), df["track_id"].astype(int)))
    if keys != sorted(keys):
        _fail(kind, "rows must be sorted by frame, then track_id")
    return df


def validate_detections(df: "pd.DataFrame", meta: dict[str, Any] | None = None) -> "pd.DataFrame":
    """Validate a Detection table (Section 4) and, if given, its consistency with the run metadata."""
    kind = "detections"
    _check_columns(df, kind)
    if df.empty:
        return df
    _check_no_nan(df, kind)
    _check_integer_column(df, "frame", kind, minimum=0)
    _check_boxes(df, kind, _image_size_from_meta(meta))
    scores = _numeric(df, ["score"], kind)["score"]
    _check_unit_interval(scores, kind, "score")
    if meta is not None:
        floor = meta.get("score_floor")
        if floor is not None and (scores < float(floor) - 1e-9).any():
            _fail(kind, f"scores below the metadata score_floor {floor} (write every detection >= the floor only)")
        _check_frames_in_frameset(df, meta, kind)
    return df


def validate_proposals(df: "pd.DataFrame", meta: dict[str, Any] | None = None) -> "pd.DataFrame":
    """Validate a Proposal table (Section 5)."""
    kind = "proposals"
    _check_columns(df, kind)
    if df.empty:
        return df
    _check_no_nan(df, kind)
    _check_integer_column(df, "frame", kind, minimum=0)
    _check_integer_column(df, "area", kind, minimum=0)
    _check_boxes(df, kind, _image_size_from_meta(meta))
    fill = _numeric(df, ["fill_ratio"], kind)["fill_ratio"]
    if (fill <= 0).any() or (fill > 1 + 1e-4).any():
        _fail(kind, "fill_ratio must lie in (0, 1]")
    scores = _numeric(df, ["score"], kind)["score"]
    unscored = scores == -1.0
    if ((scores[~unscored] < 0) | (scores[~unscored] > 1)).any():
        _fail(kind, "score must lie in [0, 1], or be -1.0 for an unscored proposal")
    if meta is not None:
        _check_frames_in_frameset(df, meta, kind)
    return df


def validate_patch_set(arrays: dict[str, "np.ndarray"]) -> None:
    """Validate a loaded patch set (Section 6): keys, dtypes, shapes, labels, kinds."""
    import json

    import numpy as np

    what = "patch_set"
    missing = [k for k in PATCH_KEYS if k not in arrays]
    if missing:
        _fail(what, f"missing key(s) {missing}")
    X = np.asarray(arrays["X"])
    if X.dtype != np.uint8 or X.ndim != 4 or tuple(X.shape[1:]) != PATCH_SHAPE:
        _fail(what, f"X must be uint8 [N, 64, 32, 3] RGB, got {X.dtype} {X.shape}")
    n = X.shape[0]
    expected = {"y": (np.int8, (n,)), "frame": (np.int32, (n,)), "box": (np.float32, (n, 4))}
    for key, (dtype, shape) in expected.items():
        a = np.asarray(arrays[key])
        if a.dtype != dtype or a.shape != shape:
            _fail(what, f"{key} must be {np.dtype(dtype).name} {shape}, got {a.dtype} {a.shape}")
    for key in ("match", "kind"):
        a = np.asarray(arrays[key])
        if a.dtype.kind != "U" or a.shape != (n,):
            _fail(what, f"{key} must be a unicode string array of shape ({n},), got {a.dtype} {a.shape}")
    if not np.isin(arrays["y"], (0, 1)).all():
        _fail(what, "y must contain only 0 (background) and 1 (player)")
    bad_kinds = sorted(set(np.asarray(arrays["kind"]).tolist()) - set(PATCH_KINDS))
    if bad_kinds:
        _fail(what, f"unknown patch kind(s) {bad_kinds}; allowed: {list(PATCH_KINDS)}")
    if n and (np.asarray(arrays["box"])[:, 2:] <= 0).any():
        _fail(what, "box width and height must be positive")
    meta_json = np.asarray(arrays["meta_json"])
    if meta_json.dtype.kind != "U" or meta_json.shape != (1,):
        _fail(what, "meta_json must be a unicode array of shape (1,)")
    try:
        info = json.loads(str(meta_json[0]))
    except json.JSONDecodeError as exc:
        _fail(what, f"meta_json is not valid JSON ({exc})")
    if info.get("schema_version") != SCHEMA_VERSION:
        _fail(what, f"meta_json schema_version must be '{SCHEMA_VERSION}'")


def validate_patch_scores(df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate a Patch score table (Section 6.1)."""
    kind = "patch_scores"
    _check_columns(df, kind)
    if df.empty:
        return df
    _check_no_nan(df, kind)
    _check_integer_column(df, "sample_id", kind, minimum=0)
    _check_integer_column(df, "frame", kind, minimum=0)
    if df["sample_id"].duplicated().any():
        _fail(kind, "duplicate sample_id")
    if not df["y_true"].isin([0, 1]).all():
        _fail(kind, "y_true must be 0 or 1")
    bad_kinds = sorted(set(df["kind"].astype(str)) - set(PATCH_KINDS))
    if bad_kinds:
        _fail(kind, f"unknown patch kind(s) {bad_kinds}")
    if not df["match"].astype(str).str.fullmatch(r"\d+").all():
        _fail(kind, "match must be a string of digits")
    _check_unit_interval(_numeric(df, ["score"], kind)["score"], kind, "score")
    return df


def validate_run_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Validate a run metadata sidecar (Section 7)."""
    import re

    what = "run_meta"
    missing = [k for k in RUN_META_REQUIRED_KEYS if k not in meta]
    if missing:
        _fail(what, f"missing key(s) {missing}")
    if meta["schema_version"] != SCHEMA_VERSION:
        _fail(what, f"schema_version must be '{SCHEMA_VERSION}'")
    if meta["kind"] not in ARTIFACT_KINDS:
        _fail(what, f"kind must be one of {list(ARTIFACT_KINDS)}")
    if meta["method"] not in DETECTOR_IDS:
        _fail(what, f"unknown method (detector ID) '{meta['method']}'")
    if meta["kind"] == "tracks":
        if meta["tracker"] not in TRACKER_IDS:
            _fail(what, f"a tracks artefact needs a registered tracker ID, got {meta['tracker']!r}")
    elif meta["tracker"] is not None:
        _fail(what, "tracker must be null unless kind is 'tracks'")
    if meta["split"] not in SPLITS:
        _fail(what, f"split must be one of {list(SPLITS)}")
    # A patch-score table covers a whole split, so it has no single match or frameset.
    per_clip = meta["kind"] != "patch_scores"
    if per_clip or meta["match"] is not None:
        if not isinstance(meta["match"], str) or not meta["match"].isdigit():
            _fail(what, "match must be a quoted string of digits, e.g. \"117093\"")
    if per_clip or meta["frames"] is not None:
        frames = meta["frames"]
        if not isinstance(frames, dict) or set(frames) != {"start", "stop", "stride"}:
            _fail(what, "frames must be {start, stop, stride}")
        if not all(isinstance(frames[k], int) and not isinstance(frames[k], bool) for k in frames):
            _fail(what, "frames start/stop/stride must be integers")
        if frames["start"] < 0 or frames["stop"] <= frames["start"] or frames["stride"] < 1:
            _fail(what, "frames must satisfy 0 <= start < stop and stride >= 1")
        if not isinstance(meta["frameset"], str) or not meta["frameset"]:
            _fail(what, "frameset must be a non-empty string")
    for key in ("image_width", "image_height"):
        if per_clip and (not isinstance(meta[key], int) or isinstance(meta[key], bool) or meta[key] <= 0):
            _fail(what, f"{key} must be a positive integer")
    if per_clip and (not isinstance(meta["fps"], (int, float)) or meta["fps"] <= 0):
        _fail(what, "fps must be a positive number")
    for key in ("score_floor", "operating_threshold"):
        value = meta[key]
        if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
            _fail(what, f"{key} must be in [0, 1] or null")
    if not isinstance(meta["frozen_test"], bool):
        _fail(what, "frozen_test must be true or false")
    if meta["split"] == "test" and not meta["frozen_test"]:
        _fail(what, "every test-split artefact must have frozen_test = true (Section 14)")
    if not isinstance(meta["config_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", meta["config_sha256"]):
        _fail(what, "config_sha256 must be 64 lowercase hex characters")
    if not isinstance(meta["created_utc"], str) or not re.fullmatch(_ISO_UTC, meta["created_utc"]):
        _fail(what, "created_utc must be ISO-8601 UTC, e.g. 2026-11-21T14:03:00Z")
    for key in ("runtime_seconds", "frames_per_second"):
        value = meta[key]
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            _fail(what, f"{key} must be a non-negative number or null")
    for key in ("device", "notes"):
        if not isinstance(meta[key], str):
            _fail(what, f"{key} must be a string")
    return meta


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_metric_value(key: str, value: Any, where: str) -> None:
    """Value domains of Section 1 / Section 8. None (null) means "undefined" and is always allowed."""
    import math

    if value is None:
        return
    if not _is_number(value) or math.isnan(value) or math.isinf(value):
        _fail("metrics", f"{where}.{key} must be a finite number or null, got {value!r}")
    if key in BOUNDED_METRIC_KEYS and not 0 <= value <= 1:
        _fail("metrics", f"{where}.{key} = {value} must be a fraction in [0, 1] (never a percentage)")
    if key in MOTA_KEYS and value > 1:
        _fail("metrics", f"{where}.{key} = {value} must be <= 1 (it may be negative)")
    if key in NONNEGATIVE_REAL_KEYS and value < 0:
        _fail("metrics", f"{where}.{key} = {value} must be >= 0")
    if key in INTEGER_METRIC_KEYS and (value < 0 or value != int(value) or isinstance(value, float)):
        _fail("metrics", f"{where}.{key} = {value} must be a non-negative integer")


def validate_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Validate a Metrics JSON document (Section 8): required keys, task-specific metric/count keys and the value
    domains (BOUNDED_METRIC_KEYS in [0, 1]; MOTA_KEYS <= 1 and possibly negative; NONNEGATIVE_REAL_KEYS >= 0;
    INTEGER_METRIC_KEYS and counts non-negative integers; null allowed for undefined values)."""
    what = "metrics"
    missing = [k for k in METRICS_REQUIRED_KEYS if k not in metrics]
    if missing:
        _fail(what, f"missing key(s) {missing}")
    if metrics["schema_version"] != SCHEMA_VERSION:
        _fail(what, f"schema_version must be '{SCHEMA_VERSION}'")
    eval_id, task = metrics["eval_id"], metrics["task"]
    if eval_id not in EVAL_TASKS:
        _fail(what, f"eval_id must be one of {list(EVAL_TASKS)}")
    if EVAL_TASKS[eval_id] != task:
        _fail(what, f"eval_id {eval_id} must have task '{EVAL_TASKS[eval_id]}', got '{task}'")
    method = metrics["method"]
    if task == "tracking":
        split_pipeline_id(method)
    elif method not in DETECTOR_IDS:
        _fail(what, f"unknown method (detector ID) '{method}'")
    if metrics["split"] not in SPLITS:
        _fail(what, f"split must be one of {list(SPLITS)}")
    match = metrics["match"]
    if not isinstance(match, str) or not (match.isdigit() or match == "pooled"):
        _fail(what, "match must be a string of digits or 'pooled'")
    expected_id = f"{eval_id}__{method}__{metrics['split']}__{match}__{metrics['frameset']}"
    if metrics["run_id"] != expected_id:
        _fail(what, f"run_id must be '{expected_id}', got '{metrics['run_id']}'")
    for section in ("params", "counts", "metrics", "inputs", "provenance"):
        if not isinstance(metrics[section], dict):
            _fail(what, f"'{section}' must be an object")
    for key in REQUIRED_METRIC_KEYS[task]:
        if key not in metrics["metrics"]:
            _fail(what, f"task '{task}' requires metrics.{key}")
    for key in REQUIRED_COUNT_KEYS[task]:
        if key not in metrics["counts"]:
            _fail(what, f"task '{task}' requires counts.{key}")
    for key, value in metrics["metrics"].items():
        _check_metric_value(key, value, "metrics")
    for key, value in metrics["counts"].items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _fail(what, f"counts.{key} = {value!r} must be a non-negative integer")
    for name, values in (metrics.get("breakdown") or {}).items():
        for label, value in (values or {}).items():
            if value is not None and (not _is_number(value) or not 0 <= value <= 1):
                _fail(what, f"breakdown.{name}.{label} = {value!r} must be in [0, 1] or null")
    for name, curve in (metrics.get("curves") or {}).items():
        lengths = {len(v) for v in curve.values()}
        if len(lengths) != 1:
            _fail(what, f"curves.{name} arrays must have the same length")
        if lengths.pop() > 200:
            _fail(what, f"curves.{name} may hold at most 200 points")
    return metrics


def validate_pitch_tracks(df: "pd.DataFrame") -> "pd.DataFrame":
    """Validate a PitchTrack table (Section 12, Phase 3)."""
    kind = "pitch_tracks"
    _check_columns(df, kind)
    if df.empty:
        return df
    _check_no_nan(df, kind)
    _check_integer_column(df, "frame", kind, minimum=0)
    _check_integer_column(df, "track_id", kind, minimum=1)
    pos = _numeric(df, ["x_m", "y_m"], kind)
    if (pos.x_m.abs() > 52.5 + 1e-6).any() or (pos.y_m.abs() > 34 + 1e-6).any():
        _fail(kind, "x_m must lie in [-52.5, 52.5] and y_m in [-34, 34] metres")
    if not df["quality"].isin([0, 1]).all():
        _fail(kind, "quality must be 0 or 1")
    if df.duplicated(["frame", "track_id"]).any():
        _fail(kind, "duplicate (frame, track_id) pairs")
    return df


def _validate_simple(df: "pd.DataFrame", kind: str) -> "pd.DataFrame":
    """Tables without a dedicated validator (workload, box_sample): columns and no missing values."""
    _check_columns(df, kind)
    if not df.empty:
        _check_no_nan(df, kind)
    return df


_VALIDATORS = {
    "tracks": validate_tracks,
    "detections": validate_detections,
    "proposals": validate_proposals,
    "patch_scores": validate_patch_scores,
    "pitch_tracks": validate_pitch_tracks,
}


def _validate_table(df: "pd.DataFrame", kind: str) -> "pd.DataFrame":
    if kind not in TABLE_SCHEMAS:
        raise ContractError(f"unknown table kind '{kind}'; registered: {list(TABLE_SCHEMAS)}")
    validator = _VALIDATORS.get(kind)
    return validator(df) if validator else _validate_simple(df, kind)


# ---- I/O (the only sanctioned way to read/write contract tables) ------------------------------------------------
def _column_dtypes(kind: str) -> dict[str, str]:
    dtypes = {}
    for c in TABLE_SCHEMAS[kind]:
        if c in _INT_COLUMNS:
            dtypes[c] = "int64"
        elif c in _STR_COLUMNS:
            dtypes[c] = "str"
        else:
            dtypes[c] = "float64"
    return dtypes


def read_table(path: Path, kind: str) -> "pd.DataFrame":
    """Read a contract CSV of `kind` (key of TABLE_SCHEMAS) with correct dtypes, then validate it."""
    import pandas as pd

    if kind not in TABLE_SCHEMAS:
        raise ContractError(f"unknown table kind '{kind}'")
    path = Path(path)
    header = pd.read_csv(path, nrows=0, encoding="utf-8")
    _check_columns(header, kind)
    df = pd.read_csv(path, dtype={c: ("str" if t == "str" else "float64") for c, t in _column_dtypes(kind).items()},
                     encoding="utf-8", keep_default_na=False, na_values=[""])
    _check_no_nan(df, kind)
    for c, t in _column_dtypes(kind).items():
        if t == "int64":
            _check_integer_column(df, c, kind)
            df[c] = df[c].astype("int64")
    return _validate_table(df, kind)


def _format_value(column: str, value: Any) -> str:
    if column in _INT_COLUMNS:
        return str(int(value))
    if column in _STR_COLUMNS:
        return str(value)
    return f"{float(value):.{_DECIMALS.get(column, _DEFAULT_DECIMALS)}f}"


def write_table(df: "pd.DataFrame", path: Path, kind: str) -> Path:
    """Validate, sort and write a contract CSV (boxes 2 decimals, scores 4 decimals, no index)."""
    if kind not in TABLE_SCHEMAS:
        raise ContractError(f"unknown table kind '{kind}'")
    _check_columns(df, kind)
    columns = TABLE_SCHEMAS[kind]
    out = df.copy()
    if not out.empty:
        _check_no_nan(out, kind)
        for c, t in _column_dtypes(kind).items():
            if t == "int64":
                _check_integer_column(out, c, kind)
                out[c] = out[c].astype("int64")
            elif t == "float64":
                out[c] = out[c].astype(float).round(_DECIMALS.get(c, _DEFAULT_DECIMALS))
            else:
                out[c] = out[c].astype(str)
        by, ascending = _SORT_KEYS[kind]
        out = out.sort_values(by, ascending=ascending, kind="mergesort").reset_index(drop=True)
    _validate_table(out, kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(columns)]
    for row in out.itertuples(index=False):
        lines.append(",".join(_format_value(c, v) for c, v in zip(columns, row)))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return path


def meta_path(table_path: Path) -> Path:
    """`<name>.csv` -> `<name>.meta.json` (Section 7: same name, next to the table)."""
    return Path(table_path).with_suffix(".meta.json")


def read_run_meta(table_path: Path) -> dict[str, Any]:
    """Read and validate the `.meta.json` sidecar that belongs to `table_path`."""
    import json

    path = meta_path(table_path)
    if not path.exists():
        raise ContractError(f"run_meta: sidecar {path.name} not found next to {Path(table_path).name}")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    return validate_run_meta(meta)


def write_run_meta(meta: dict[str, Any], table_path: Path) -> Path:
    """Validate and write the `.meta.json` sidecar next to `table_path`."""
    import json

    validate_run_meta(meta)
    path = meta_path(table_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path
