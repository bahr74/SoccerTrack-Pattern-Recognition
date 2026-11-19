"""Tests owned by Member 2. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 2, M1): make_tiles covers a 4096x1080 frame completely with at least overlap_px overlap and the last tile flush with the edge")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 2, M1): make_tiles handles 3840x1504 (likely 132831 geometry) and frames smaller than one tile")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 2, M1): tile_to_frame places tile-local boxes in full-frame coordinates (round trip)")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 2, M1): merge_tile_detections keeps one box for a player duplicated across a tile border")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 2, M3): apply_pitch_filter returns the input unchanged with note pitch_filter=off when the mask file is missing")
def test_05(): ...


@pytest.mark.skip(reason="TODO(Member 2, M3): run_detection output (mocked model) passes contracts.validate_detections and keeps scores >= conf_floor")
def test_06(): ...


@pytest.mark.gpu
@pytest.mark.skip(reason="TODO(Member 2, M2): YoloTiledDetector on one real 117093 frame returns more than 10 person boxes in full-frame coordinates")
def test_gpu_01(): ...
