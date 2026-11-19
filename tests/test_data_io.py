"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M1): read_mot_raw parses tests/fixtures/mot_raw_sample.txt: 9 rows, 10 columns, frames start at 0")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): mot_to_tracks on mot_raw_sample.txt equals tests/fixtures/tracks_gt.csv exactly")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): mot_to_tracks on mot_raw_dirty.txt (expected 3 per frame, 640x360) gives 11 rows; issues degenerate 1, clipped 1, duplicate 1, frames_not_expected 1")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): iter_frames on a tiny synthetic video (written to tmp_path with cv2.VideoWriter) yields 0-based indices in order and honours frame_ids")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 1, M2): compute_eda_stats on fixtures produces a dict with the structure of eda_stats_sample.json")
def test_05(): ...


@pytest.mark.data
@pytest.mark.skip(reason="TODO(Member 1, M1): real data: decoded frame count == max(frame)+1 for every local train/val match")
def test_data_01(): ...
