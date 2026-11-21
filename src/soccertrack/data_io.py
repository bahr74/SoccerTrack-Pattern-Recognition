"""Dataset access: raw MOT parsing, cleaning, sequential video decoding and EDA statistics.

Owner: Member 1. Priority milestone M1 (day 1): `read_mot_raw`, `mot_to_tracks`, `video_info`, `iter_frames`
are used by Members 2, 3 and 4. Until they exist, other members use tests/fixtures/.
Contracts: SHARED_CONTRACTS.md Sections 1, 3, 3.1 and 11.

Facts verified on the real files (2026-11-19): 10 columns, 0-based frames, columns 7-10 are -1 in every row,
22 boxes per frame, 22 IDs per clip. Clip 117093 has a 0.979 s container start offset -> decode by index only.
"""

from __future__ import annotations

from numbers import Integral
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

from soccertrack import config, contracts

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


_RAW_COLUMNS = ("frame", "id", "x", "y", "w", "h", "c7", "c8", "c9", "c10")


def _typed_raw(raw: "pd.DataFrame") -> "pd.DataFrame":
    """Check raw numeric fields without changing rows or box coordinates."""
    import numpy as np
    import pandas as pd

    if tuple(raw.columns) != _RAW_COLUMNS:
        raise ValueError("MOT input must have exactly 10 columns: " + ",".join(_RAW_COLUMNS))
    result = raw.copy()
    for name in _RAW_COLUMNS[2:]:
        result[name] = pd.to_numeric(raw[name], errors="raise").astype("float64")
    for name, minimum in (("frame", 0), ("id", 1)):
        integers = []
        for value in raw[name]:
            try:
                exact = Decimal(str(value))
            except InvalidOperation as exc:
                raise ValueError(f"MOT {name} must contain integers >= {minimum}") from exc
            if (
                isinstance(value, bool) or not exact.is_finite()
                or exact < minimum or exact >= 2**63
                or exact != exact.to_integral_value()
            ):
                raise ValueError(f"MOT {name} must contain int64 integers >= {minimum}")
            integers.append(int(exact))
        result[name] = pd.Series(integers, index=raw.index, dtype="int64")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("MOT fields must be finite numbers")
    return result


def read_mot_raw(path: Path) -> "pd.DataFrame":
    """Read a raw SoccerTrack MOT txt file into columns frame,id,x,y,w,h,c7,c8,c9,c10 (no cleaning).

    Frames/IDs are int64; boxes and unused fields are float64. Reject malformed,
    missing or non-finite values. An empty file returns the same typed schema.
    """
    import pandas as pd

    try:
        # Preserve the original integer tokens before pandas can round via float64.
        raw = pd.read_csv(path, header=None, dtype=str)
    except pd.errors.EmptyDataError:
        raw = pd.DataFrame(columns=_RAW_COLUMNS)
    if raw.shape[1] != len(_RAW_COLUMNS):
        raise ValueError("MOT input must have exactly 10 columns")
    raw.columns = _RAW_COLUMNS
    return _typed_raw(raw)


def clean_mot_geometry(
    raw: "pd.DataFrame", image_size: tuple[int, int], expected_boxes_per_frame: int = 22
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Pure Member 1 geometry cleaning, before shared contract certification.

    Returns (tracks, issues) where issues has the keys degenerate_boxes, clipped_boxes, duplicate_rows,
    frames_not_expected (counted after de-duplication and degenerate-box removal).
    image_size is (W, H) in full-frame pixels. Count each removed degenerate row
    once, including boxes that become degenerate after clipping. Missing frame
    indices in 0..max(raw.frame) count as frames_not_expected, including frames
    emptied by cleaning. Conflicting (frame, id) rows are rejected, not merged.
    This local result is NOT certified by Member 3. Production consumers use
    mot_to_tracks, which adds the required shared validator without a fallback.
    Fixture check: tests/fixtures/mot_raw_dirty.txt -> issues {1, 1, 1, 1}, 11 rows.
    """
    if len(image_size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, Integral) or v <= 1 for v in image_size
    ):
        raise ValueError("image_size must be positive integer (W, H), both > 1")
    if (
        isinstance(expected_boxes_per_frame, bool)
        or not isinstance(expected_boxes_per_frame, Integral)
        or expected_boxes_per_frame < 1
    ):
        raise ValueError("expected_boxes_per_frame must be a positive integer")
    raw = _typed_raw(raw)
    cleaned = raw.drop_duplicates().copy()
    issues = {"duplicate_rows": len(raw) - len(cleaned)}
    degenerate = (cleaned.w <= 1) | (cleaned.h <= 1)
    issues["degenerate_boxes"] = int(degenerate.sum())
    cleaned = cleaned.loc[~degenerate].copy()
    width, height = image_size
    x0, y0 = cleaned.x.clip(0, width), cleaned.y.clip(0, height)
    x1 = (cleaned.x + cleaned.w).clip(0, width)
    y1 = (cleaned.y + cleaned.h).clip(0, height)
    clipped = (
        (x0 != cleaned.x) | (y0 != cleaned.y)
        | (x1 != cleaned.x + cleaned.w) | (y1 != cleaned.y + cleaned.h)
    )
    issues["clipped_boxes"] = int(clipped.sum())
    cleaned = cleaned.assign(x=x0, y=y0, w=x1 - x0, h=y1 - y0)
    degenerate = (cleaned.w <= 1) | (cleaned.h <= 1)
    issues["degenerate_boxes"] += int(degenerate.sum())
    cleaned = cleaned.loc[~degenerate]
    if cleaned.duplicated(["frame", "id"]).any():
        raise ValueError("Conflicting MOT rows for the same (frame, id)")
    counts = cleaned.groupby("frame").size()
    total_frames = int(raw.frame.max()) + 1 if len(raw) else 0
    issues["frames_not_expected"] = total_frames - int((counts == expected_boxes_per_frame).sum())
    tracks = cleaned.rename(columns={"id": "track_id"}).assign(score=1.0)
    tracks = (
        tracks.loc[:, list(contracts.TRACK_COLUMNS)]
        .sort_values(["frame", "track_id"]).reset_index(drop=True)
    )
    return tracks, issues


def mot_to_tracks(
    raw: "pd.DataFrame", image_size: tuple[int, int], expected_boxes_per_frame: int = 22
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Clean MOT geometry and require official shared Track-table validation."""
    tracks, issues = clean_mot_geometry(raw, image_size, expected_boxes_per_frame)
    contracts.validate_tracks(tracks, image_size=image_size)
    return tracks, issues


def load_gt_tracks(match_id: str, sources: dict[str, Any]) -> "pd.DataFrame":
    """Return the cleaned GT Track table for a match (uses config.match_paths and video_info)."""
    paths = config.match_paths(match_id, sources)
    info = video_info(paths.clip)
    tracks, _ = mot_to_tracks(read_mot_raw(paths.mot), (info["width"], info["height"]))
    return tracks


def video_info(clip_path: Path) -> dict[str, Any]:
    """Return {"width", "height", "fps", "frame_count_header"} from the container (header values only).

    The authoritative frame count is the sequentially decoded count (`count_decoded_frames`).
    """
    import cv2

    capture = _open_video(clip_path)
    try:
        return {
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "frame_count_header": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
    finally:
        capture.release()


def count_decoded_frames(clip_path: Path) -> int:
    """Decode the whole clip and count successful images; callers should cache this slow result."""
    return sum(1 for _ in iter_frames(clip_path))


def _open_video(clip_path: Path) -> Any:
    """Open a local clip; release even an unsuccessful capture."""
    import cv2

    if not Path(clip_path).is_file():
        raise FileNotFoundError(clip_path)
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Cannot open video: {clip_path}")
    return capture


def iter_frames(clip_path: Path, frame_ids: Iterable[int] | None = None) -> Iterator[tuple[int, "np.ndarray"]]:
    """Yield (frame_index, BGR uint8 image) by SEQUENTIAL decoding; skip frames not in `frame_ids`.

    Never seek by timestamp. Frame indices are 0-based and equal to the MOT `frame` column.
    Requested IDs may be unordered or repeated; yield each available ID once in
    clip order. IDs beyond EOF yield nothing. Negative/non-integer IDs raise.
    Empty selections do not open the clip. Release on exhaustion, error or close().
    """
    requested = None if frame_ids is None else list(frame_ids)
    selected = None
    if requested is not None:
        if any(isinstance(i, bool) or not isinstance(i, Integral) or i < 0 for i in requested):
            raise ValueError("frame_ids must contain non-negative integers")
        selected = set(requested)
        if not selected:
            return
    last = max(selected) if selected is not None else None
    capture = _open_video(clip_path)
    try:
        index = 0
        while (last is None or index <= last) and capture.grab():
            if selected is None or index in selected:
                ok, image = capture.retrieve()
                if not ok or image is None:
                    raise OSError(f"Could not decode frame {index} of {clip_path}")
                yield index, image
            index += 1
    finally:
        capture.release()


def compute_eda_stats(splits: dict[str, list[str]], sources: dict[str, Any]) -> dict[str, Any]:
    """Return Contracts section 11 statistics for configured train/val matches.

    Refuse held-out inputs before resolving paths. Unconfigured matches are not
    available on this machine; configured missing files are errors. Box and track
    summaries use cleaned annotations. Annotated frame counts include raw frames
    emptied by cleaning; box-count statistics include those zero-count frames.
    Additional unannotated decoded frames contribute to frames_not_expected.
    No dataset files or outputs are written by this function.
    """
    from soccertrack import split as match_split

    if set(splits) - {"train", "val", "test"}:
        raise ValueError("EDA splits must be train or val")
    if splits.get("test"):
        raise PermissionError("EDA must not access held-out test matches")
    split_cfg = {"split": {name: splits.get(name, []) for name in ("train", "val", "test")}}
    match_split.assert_disjoint(split_cfg)
    cfg = config.load_config()
    project = cfg["project"]
    expected = project["dataset"]["expected_boxes_per_frame"]
    for name in ("train", "val"):
        for match in splits.get(name, []):
            actual = match_split.split_of(match, cfg)
            if actual == "test":
                raise PermissionError(f"EDA must not access held-out match {match}")
            if actual != name:
                raise ValueError(f"Match {match} belongs to {actual}, not {name}")
    result: dict[str, Any] = {
        "schema_version": contracts.SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "matches": {},
        "totals": {"matches": 0, "frames": 0, "boxes": 0},
        "paper_reference": {"frames": 59842, "boxes": 1316524, "median_box_wh_px": [25, 55]},
    }
    for name in ("train", "val"):
        for match in splits.get(name, []):
            if match not in sources["matches"]:
                continue
            paths = config.match_paths(match, sources)
            missing = [str(p) for p in (paths.mot, paths.clip) if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Configured {name} match {match}: missing files: {missing}")
            raw = read_mot_raw(paths.mot)
            if raw.empty:
                raise ValueError(f"Match {match} has no annotations")
            info = video_info(paths.clip)
            tracks, issues = mot_to_tracks(raw, (info["width"], info["height"]), expected)
            if tracks.empty:
                raise ValueError(f"Match {match} has no valid boxes after cleaning")
            decoded = count_decoded_frames(paths.clip)
            if decoded == 0:
                raise ValueError(f"Match {match} has no decodable frames")
            annotated = raw.frame.nunique()
            counts = tracks.groupby("frame").size().reindex(raw.frame.unique(), fill_value=0)
            lengths = tracks.groupby("track_id").size()
            issues["frames_not_expected"] += max(0, decoded - (int(raw.frame.max()) + 1))
            result["matches"][match] = {
                "split": name, "image_width": info["width"], "image_height": info["height"],
                "fps": info["fps"], "frames_annotated": int(annotated),
                "frames_decoded": decoded, "boxes": len(tracks), "ids": int(tracks.track_id.nunique()),
                "boxes_per_frame": {"min": int(counts.min()), "max": int(counts.max()), "mean": float(counts.mean())},
                "box_width_px": _box_quantiles(tracks.w),
                "box_height_px": _box_quantiles(tracks.h),
                "track_length_frames": {"min": int(lengths.min()), "median": float(lengths.median()), "max": int(lengths.max())},
                "issues": issues,
            }
            result["totals"]["matches"] += 1
            result["totals"]["frames"] += int(annotated)
            result["totals"]["boxes"] += len(tracks)
    return result


def _box_quantiles(values: "pd.Series") -> dict[str, float]:
    """Linear-interpolated percentiles of cleaned full-frame box dimensions."""
    return {name: float(values.quantile(q, interpolation="linear"))
            for name, q in (("p05", 0.05), ("median", 0.5), ("p95", 0.95))}
