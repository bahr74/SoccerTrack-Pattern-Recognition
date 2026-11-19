"""Tests owned by Member 4. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 4, Phase 3): fit_mapping recovers a known synthetic homography; leave_one_out_error is near 0 for exact points")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 4, Phase 3): pitch_tracks_sample.csv: distance 4.8 m over 25 frames, speed 5.0 m/s, no sprint distance")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 4, Phase 3): summarise output has exactly the workload contract columns")
def test_03(): ...
