"""Tests owned by Member 4. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# test_01 — draw_boxes: returns same shape, does not modify input
# ---------------------------------------------------------------------------
def test_01(synthetic_frame):
    """draw_boxes on synthetic_frame + tracks_gt.csv returns an image of the same shape
    and does not modify the input frame."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from soccertrack.visualize import draw_boxes

    # Load fixture tracks
    tracks = pd.read_csv(FIXTURES / "tracks_gt.csv")

    # Build a synthetic frame with player-sized boxes at fixture positions
    frame_id = 0
    rows = tracks[tracks["frame"] == frame_id]
    boxes = [(r["x"], r["y"], r["w"], r["h"]) for _, r in rows.iterrows()]
    frame = synthetic_frame(boxes=boxes, size=(640, 360))

    original = frame.copy()

    # Call draw_boxes
    result = draw_boxes(frame, tracks, frame_id=0)

    # Must return a numpy array of same shape
    assert isinstance(result, np.ndarray), "draw_boxes must return a numpy array"
    assert result.shape == frame.shape, (
        f"Output shape {result.shape} != input shape {frame.shape}"
    )
    assert result.dtype == np.uint8, "Output must be uint8"

    # Input must not be modified (draw_boxes works on a copy)
    assert np.array_equal(frame, original), "draw_boxes must not modify the input frame"

    # At least one pixel must differ from the pure frame (boxes were drawn)
    assert not np.array_equal(result, original), "draw_boxes produced no visible change"


# ---------------------------------------------------------------------------
# test_02 — plot_eda: writes expected figure files to tmp_path
# ---------------------------------------------------------------------------
def test_02(tmp_path):
    """plot_eda on tests/fixtures/eda_stats_sample.json writes the expected figure files."""
    json_mod = pytest.importorskip("json")
    pytest.importorskip("matplotlib")
    from soccertrack.visualize import plot_eda

    with (FIXTURES / "eda_stats_sample.json").open("r", encoding="utf-8") as fh:
        stats = json.load(fh)

    produced = plot_eda(stats, box_sample=None, out_dir=tmp_path)

    # Must return a list of Path objects
    assert isinstance(produced, list), "plot_eda must return a list"
    assert len(produced) >= 4, f"Expected at least 4 figures, got {len(produced)}"

    expected_names = {
        "boxes_per_match.png",
        "box_size_hist.png",
        "track_length.png",
        "data_issues.png",
    }
    written_names = {p.name for p in produced}
    for name in expected_names:
        assert name in written_names, f"Missing expected figure: {name}"
        assert (tmp_path / name).exists(), f"Figure file not on disk: {name}"
        assert (tmp_path / name).stat().st_size > 0, f"Figure file is empty: {name}"


# ---------------------------------------------------------------------------
# test_03 — side_by_side: stacks two overlays with titles
# ---------------------------------------------------------------------------
def test_03(synthetic_frame):
    """side_by_side stacks two overlays vertically and returns a taller image."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from soccertrack.visualize import side_by_side

    tracks_gt = pd.read_csv(FIXTURES / "tracks_gt.csv")
    tracks_pred = pd.read_csv(FIXTURES / "tracks_pred.csv")

    frame = synthetic_frame(size=(640, 360))
    result = side_by_side(
        frame,
        left=tracks_gt,
        right=tracks_pred,
        frame_id=0,
        titles=("Pipeline A (GT)", "Pipeline B (Pred)"),
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8

    # Vertically stacked: height must be 2× the original (allow ±2 for rounding)
    assert abs(result.shape[0] - 2 * frame.shape[0]) <= 2, (
        f"Expected stacked height ~{2 * frame.shape[0]}, got {result.shape[0]}"
    )
    # Width must match the original
    assert result.shape[1] == frame.shape[1], "Width must not change"
    # 3 channels
    assert result.shape[2] == 3


# ---------------------------------------------------------------------------
# test_04 — plot_stage_figure: composes 2x2 grid (with placeholders if missing)
# ---------------------------------------------------------------------------
def test_04_plot_stage_figure(tmp_path):
    """plot_stage_figure writes a PNG even when some stage images are placeholders."""
    pytest.importorskip("matplotlib")
    from soccertrack.visualize import plot_stage_figure

    out_png = tmp_path / "stage_0.png"
    ret = plot_stage_figure(tmp_path / "missing_stages", frame_id=0, out_path=out_png)
    assert ret == out_png
    assert out_png.exists() and out_png.stat().st_size > 0
