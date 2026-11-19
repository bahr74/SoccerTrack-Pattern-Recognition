"""Tests owned by Member 2. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 2, M4): DetectionsForByteTrack exposes conf, cls, xywh (centre format) and supports boolean indexing")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 2, M4): track_sequence on tracks_gt.csv used as detections (gt_oracle) gives no ID switches")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 2, M4): frames without detections still advance the tracker; output passes contracts.validate_tracks")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 2, M4): output_box=detection writes the raw detection box selected through the returned idx")
def test_04(): ...
