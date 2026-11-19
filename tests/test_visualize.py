"""Tests owned by Member 4. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 4, M1): draw_boxes on synthetic_frame + tracks_gt.csv returns an image of the same shape and does not modify the input")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 4, M1): plot_eda on tests/fixtures/eda_stats_sample.json writes the expected figure files to tmp_path")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 4, M2): side_by_side stacks two overlays with titles")
def test_03(): ...
