"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M3): pitch_mask on a synthetic green frame (conftest.synthetic_frame) covers the green area")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M3): foreground_mask + clean_mask isolate the dark rectangles of synthetic_frame(boxes of tracks_gt.csv frame 0)")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M3): propose returns full-resolution boxes with IoU >= 0.8 to the synthetic boxes, also when work_scale = 0.5")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M3): fit_size_model recovers slope and intercept from synthetic boxes whose height grows linearly with y")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 1, M3): proposal_recall on tests/fixtures/proposals_sample.csv vs tracks_gt.csv matches a hand-computed value")
def test_05(): ...


@pytest.mark.data
@pytest.mark.skip(reason="TODO(Member 1, M3): real data: proposal recall at IoU 0.5 on 117093 eval10 is reported (no threshold asserted)")
def test_data_01(): ...
