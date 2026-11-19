"""Shared pytest fixtures. Owner: Member 3 (test infrastructure). Other members add fixtures in their own test files.

Everything here is synthetic and small. No real data, no network, no GPU.
Fixture semantics and hand-computed expectations: docs/team/SHARED_CONTRACTS.md, Section 15.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE_IMAGE_SIZE = (640, 360)  # (W, H)
FIXTURE_MATCH = "999001"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Folder holding the committed synthetic contract fixtures (read-only)."""
    return FIXTURES


@pytest.fixture()
def synthetic_frame():
    """Factory: BGR uint8 frame (H, W, 3) with a green pitch and dark player rectangles at the given xywh boxes."""
    np = pytest.importorskip("numpy")

    def make(boxes=(), size=FIXTURE_IMAGE_SIZE, pitch_bgr=(40, 140, 40), player_bgr=(30, 30, 160)):
        width, height = size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = pitch_bgr
        for x, y, w, h in boxes:
            x0, y0 = int(round(x)), int(round(y))
            frame[y0:y0 + int(round(h)), x0:x0 + int(round(w))] = player_bgr
        return frame

    return make


@pytest.fixture()
def synthetic_patches():
    """Factory: (X uint8 [N, 64, 32, 3] RGB, y int8 [N]); positives contain a dark vertical blob, negatives are grass."""
    np = pytest.importorskip("numpy")

    def make(n_pos=8, n_neg=24, seed=0):
        rng = np.random.default_rng(seed)
        X = np.empty((n_pos + n_neg, 64, 32, 3), dtype=np.uint8)
        X[:] = (40, 140, 40)
        X = np.clip(X.astype(np.int16) + rng.integers(-10, 11, X.shape), 0, 255).astype(np.uint8)
        X[:n_pos, 8:58, 10:22] = (160, 30, 30)
        y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int8)
        return X, y

    return make
