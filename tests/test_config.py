"""Tests owned by Member 4. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 4, M0): load_config returns the four sections; split lists are disjoint")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 4, M0): load_data_sources rejects unfilled placeholders (<ST_LOCAL>) with a helpful message")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 4, M0): artifact_path builds outputs/detections/<det>/<split>/<match>__<frameset>.csv and refuses read-only roots")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 4, M0): frameset_range(eval10, 5976) == range(0, 5976, 10); win60s is clamped to the clip length")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 4, M0): config_sha256 is stable across line-ending changes and ignores data_sources.yaml")
def test_05(): ...


@pytest.mark.data
@pytest.mark.skip(reason="TODO(Member 4, M0): every train/val and weights path in configs/data_sources.yaml exists (read-only check)")
def test_data_01(): ...
