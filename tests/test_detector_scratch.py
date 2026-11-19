"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M3): rule_scores returns values in [0, 1] and ranks the player-shaped proposal (proposals_sample.csv row 0) above the line-shaped one (row 4)")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M3): run_detection output on a synthetic clip passes contracts.validate_detections and has a valid .meta.json")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M7): NMS removes duplicate boxes (IoU > nms_iou) and keeps the higher score")
def test_03(): ...
