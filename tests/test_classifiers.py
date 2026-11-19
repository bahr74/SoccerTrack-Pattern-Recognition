"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="TODO(Member 1, M5): HogSvmClassifier separates conftest.synthetic_patches (train accuracy > 0.95) and predict_proba is in [0, 1]")
def test_01(): ...


@pytest.mark.skip(reason="TODO(Member 1, M5): HogSvmClassifier save/load round trip gives identical scores; hog_weights has shape (756,)")
def test_02(): ...


@pytest.mark.skip(reason="TODO(Member 1, M6): CnnClassifier has fewer than 100k parameters and outputs shape [N]")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 1, M6): CnnClassifier overfits one small synthetic batch (loss decreases below 0.1)")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 1, M6): CnnClassifier exposes model, gradcam_layer and preprocess (interface used by Member 3)")
def test_05(): ...
