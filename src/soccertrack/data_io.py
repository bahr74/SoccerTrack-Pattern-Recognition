"""Dataset access: raw MOT parsing, cleaning, sequential video decoding and EDA statistics.

Owner: Member 1. Priority milestone M1 (day 1): `read_mot_raw`, `mot_to_tracks`, `video_info`, `iter_frames`
are used by Members 2, 3 and 4. Until they exist, other members use tests/fixtures/.
Contracts: SHARED_CONTRACTS.md Sections 1, 3, 3.1 and 11.

Facts verified on the real files (2026-11-19): 10 columns, 0-based frames, columns 7-10 are -1 in every row,
22 boxes per frame, 22 IDs per clip. Clip 117093 has a 0.979 s container start offset -> decode by index only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def read_mot_raw(path: Path) -> "pd.DataFrame":
    """Read a raw SoccerTrack MOT txt file into columns frame,id,x,y,w,h,c7,c8,c9,c10 (no cleaning).

    TODO(Member 1, M1): assert exactly 10 columns; integer frame/id; float boxes.
    """
    raise NotImplementedError("TODO(Member 1, M1)")


def mot_to_tracks(
    raw: "pd.DataFrame", image_size: tuple[int, int], expected_boxes_per_frame: int = 22
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Clean a raw MOT table and convert it to the Track table contract.

    Returns (tracks, issues) where issues has the keys degenerate_boxes, clipped_boxes, duplicate_rows,
    frames_not_expected (counted after de-duplication and degenerate-box removal).
    TODO(Member 1, M1): drop exact duplicates, drop boxes with w<=1 or h<=1, clip boxes to the image,
    set track_id=id and score=1.0, validate with contracts.validate_tracks.
    Fixture check: tests/fixtures/mot_raw_dirty.txt -> issues {1, 1, 1, 1}, 11 rows.
    """
    raise NotImplementedError("TODO(Member 1, M1)")


def load_gt_tracks(match_id: str, sources: dict[str, Any]) -> "pd.DataFrame":
    """Return the cleaned GT Track table for a match (uses config.match_paths and video_info)."""
    raise NotImplementedError("TODO(Member 1, M1)")


def video_info(clip_path: Path) -> dict[str, Any]:
    """Return {"width", "height", "fps", "frame_count_header"} from the container (header values only).

    The authoritative frame count is the sequentially decoded count (`count_decoded_frames`).
    """
    raise NotImplementedError("TODO(Member 1, M1)")


def count_decoded_frames(clip_path: Path) -> int:
    """Decode the whole clip sequentially and return the number of frames (slow; cache the result)."""
    raise NotImplementedError("TODO(Member 1, M1)")


def iter_frames(clip_path: Path, frame_ids: Iterable[int] | None = None) -> Iterator[tuple[int, "np.ndarray"]]:
    """Yield (frame_index, BGR uint8 image) by SEQUENTIAL decoding; skip frames not in `frame_ids`.

    Never seek by timestamp. Frame indices are 0-based and equal to the MOT `frame` column.
    TODO(Member 1, M1): use cv2.VideoCapture + grab()/retrieve() so skipped frames are not fully decoded.
    """
    raise NotImplementedError("TODO(Member 1, M1)")


def compute_eda_stats(splits: dict[str, list[str]], sources: dict[str, Any]) -> dict[str, Any]:
    """Compute outputs/eda/stats.json content (SHARED_CONTRACTS.md Section 11) for all available matches."""
    raise NotImplementedError("TODO(Member 1, M2)")
