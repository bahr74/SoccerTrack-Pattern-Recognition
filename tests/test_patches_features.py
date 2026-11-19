"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M4): crop_patch returns uint8 RGB (64, 32, 3), pads at image borders, converts BGR to RGB")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M4): negative sampling respects IoU bounds (random < 0.1, partial in [0.1, 0.4)) and the 1:3 ratio")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M4): saved patch set passes contracts.validate_patch_set; no test-split matches unless frozen_test")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M5): hog_features returns float32 [N, 756]")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 1, M5): hog_features agrees with cv2.HOGDescriptor((32,64),(16,16),(8,8),(8,8),9) within a documented tolerance")
def test_05(): ...
