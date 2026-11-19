"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M8): on tracks_gt.csv used as detections (gt_oracle), every GT ID maps to exactly one track ID (no ID switch)")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M8): two synthetic objects crossing horizontally keep their IDs")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M8): a track is confirmed only after n_init hits and deleted after max_age missed frames")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M8): output passes contracts.validate_tracks; the module never imports tracker_bytetrack")
def test_04(): ...
