"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 3, M0): iou_matrix: identical boxes give 1, disjoint boxes 0, (500,140,18,40) vs (510,150,18,40) gives 0.2")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): detections_sample.csv vs tracks_gt.csv at IoU 0.5 and threshold 0.5: TP 7, FP 1, FN 2, P 0.8750, R 0.7778, F1 0.8235")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): all-point AP at IoU 0.5 over all fixture detections = 0.8667 (TP 8, FP 3)")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): patch_scores_sample.csv at threshold 0.5: TP 3, FN 1, TN 3, FP 1; ROC-AUC 0.9375")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): own AP agrees with pycocotools within 0.01 on a random synthetic set")
def test_05(): ...
