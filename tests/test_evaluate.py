"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 3, M1): evaluate_detections on the fixtures returns a dict that passes contracts.validate_metrics")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): evaluate_patches and evaluate_proposals produce E1 and E2 Metrics JSON from the fixtures")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 3, M2): gt_as_detections and noisy_gt_detections produce valid Detection tables; noisy drop rate within tolerance")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): export_trackeval(tracks_gt.csv, tracks_pred.csv) reproduces tests/fixtures/trackeval_example byte for byte")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): GT evaluated against itself gives HOTA = IDF1 = MOTA = 1.0")
def test_05(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): tracks_pred.csv gives MOTA 0.6667, IDF1 0.7500, IDSW 1")
def test_06(): ...


@pytest.mark.skip(reason="TODO(Member 3, M4): pool_metrics recomputes P, R and F1 from summed counts")
def test_07(): ...
