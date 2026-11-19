"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M1): matches_for returns the approved split from configs/project.yaml; optional matches only when enabled")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): assert_disjoint raises when a match is in two splits")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): require_test_access raises for split=test without frozen_test and passes with it")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M1): log_test_access appends an entry with config_sha256 to a temporary test log")
def test_04(): ...
