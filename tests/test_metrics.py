"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


def test_01():
    np = pytest.importorskip("numpy")
    from soccertrack.metrics import iou_matrix

    a = np.array([[500, 140, 18, 40]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[0, 0, 10, 10]]))[0, 0] == 0.0
    # Overlap 8 x 30 = 240; union 720 + 720 - 240 = 1200; IoU = 0.2.
    assert iou_matrix(a, np.array([[510, 150, 18, 40]]))[0, 0] == pytest.approx(0.2)
    # Touching edges share no area.
    assert iou_matrix(a, np.array([[518, 140, 18, 40]]))[0, 0] == 0.0
    # Shape [N, M] and empty inputs.
    m = iou_matrix(np.zeros((3, 4)) + [0, 0, 5, 5], np.zeros((2, 4)) + [0, 0, 5, 5])
    assert m.shape == (3, 2)
    assert iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)


@pytest.mark.skip(reason="TODO(Member 3, M1): detections_sample.csv vs tracks_gt.csv at IoU 0.5 and threshold 0.5: TP 7, FP 1, FN 2, P 0.8750, R 0.7778, F1 0.8235")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): all-point AP at IoU 0.5 over all fixture detections = 0.8667 (TP 8, FP 3)")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): patch_scores_sample.csv at threshold 0.5: TP 3, FN 1, TN 3, FP 1; ROC-AUC 0.9375")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 3, M1): own AP agrees with pycocotools within 0.01 on a random synthetic set")
def test_05(): ...
