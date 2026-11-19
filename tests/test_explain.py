"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 3, Phase 3): gradcam returns [N, 64, 32] maps in [0, 1] for a randomly initialised CnnClassifier")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 3, Phase 3): stratify_detection_errors labels the fixture detections consistently with metrics.match_frame")
def test_02(): ...
