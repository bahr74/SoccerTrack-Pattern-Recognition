"""Tests owned by Member 1. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from soccertrack import config, proposals, segmentation


@pytest.fixture
def scratch_cfg():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/scratch.yaml").read_text(encoding="utf-8"))


def test_01(synthetic_frame, scratch_cfg):
    frame = synthetic_frame(boxes=[(100, 150, 20, 45)])
    mask = segmentation.pitch_mask(frame, scratch_cfg, "999001")
    assert mask.dtype == bool
    assert mask.shape == frame.shape[:2]
    assert mask.all()  # the player-shaped hole is filled inside the pitch


def test_pitch_largest_component_and_empty_mask(scratch_cfg):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    frame[10:100, 10:100] = (40, 140, 40)
    frame[40:50, 40:50] = 0
    frame[10:20, 130:140] = (40, 140, 40)
    mask = segmentation.pitch_mask(frame, scratch_cfg, "999001")
    assert mask[45, 45]
    assert not mask[15, 135]
    assert not mask[110, 110]
    assert not segmentation.pitch_mask(np.zeros_like(frame), scratch_cfg, "999001").any()


@pytest.mark.parametrize("colorspace", ["lab", "gray"])
def test_02(synthetic_frame, fixtures_dir, scratch_cfg, colorspace):
    scratch_cfg["segmentation"]["diff_colorspace"] = colorspace
    gt = pd.read_csv(fixtures_dir / "tracks_gt.csv")
    boxes = gt.loc[gt.frame == 0, ["x", "y", "w", "h"]].to_numpy()
    background = synthetic_frame()
    frame = synthetic_frame(boxes=boxes)
    pitch = np.ones(background.shape[:2], dtype=bool)
    fg = segmentation.foreground_mask(frame, background, pitch, scratch_cfg)
    clean = segmentation.clean_mask(fg, scratch_cfg)
    count, _, stats, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8))
    assert count - 1 == 3
    assert all(clean[int(y + h / 2), int(x + w / 2)] for x, y, w, h in boxes)
    assert not clean[20, 20]
    assert not segmentation.foreground_mask(frame, background, np.zeros_like(pitch), scratch_cfg).any()
    # Blur/threshold and morphology may expand or contract boundaries slightly.
    for actual, expected in zip(sorted(stats[1:], key=lambda row: row[0]), boxes, strict=True):
        x, y, w, h = actual[:4]
        gx, gy, gw, gh = expected
        intersection = max(0, min(x + w, gx + gw) - max(x, gx)) * max(0, min(y + h, gy + gh) - max(y, gy))
        assert intersection / (w * h + gw * gh - intersection) >= 0.8


@pytest.mark.parametrize("scale", [1., .5])
def test_03(scratch_cfg, fixtures_dir, scale):
    from soccertrack.tracker_iou import _iou
    gt = pd.read_csv(fixtures_dir / "tracks_gt.csv").query("frame == 0")
    mask = np.zeros((360,640), dtype=np.uint8)
    for x,y,w,h in gt[["x","y","w","h"]].to_numpy(int):
        mask[y:y+h,x:x+w] = 1
    small = cv2.resize(mask, (int(640*scale),int(360*scale)), interpolation=cv2.INTER_NEAREST).astype(bool)
    result = proposals.propose(small,0,scale,proposals.SizeModel(0,45,20/45),scratch_cfg)
    assert len(result) == 3
    overlap = _iou(result[["x","y","w","h"]].to_numpy(), gt[["x","y","w","h"]].to_numpy())
    assert (overlap.max(axis=0) >= .8).all()
    assert (result.score == -1).all() and (result.fill_ratio <= 1).all()


def test_proposals_empty_border_and_wide_split(scratch_cfg):
    from soccertrack import contracts
    model = proposals.SizeModel(0,40,.5)
    empty = proposals.propose(np.zeros((100,100),bool),0,1,model,scratch_cfg)
    assert empty.empty and tuple(empty.columns)==contracts.PROPOSAL_COLUMNS
    mask=np.zeros((100,100),bool); mask[0:40,0:60]=True
    result=proposals.propose(mask,0,1,model,scratch_cfg)
    assert len(result)==3
    assert result.area.sum()==2400 and (result.x+result.w<=100).all()
    with pytest.raises(ValueError): proposals.propose(mask,-1,1,model,scratch_cfg)


def test_04():
    bottom = np.array([100., 200., 300., 400.])
    heights = 0.1 * bottom + 20
    train = pd.DataFrame({"y": bottom - heights, "h": heights})
    train.attrs["split"] = "train"
    model = proposals.fit_size_model(train)
    assert model.slope == pytest.approx(0.1)
    assert model.intercept == pytest.approx(20)
    np.testing.assert_allclose(model.expected_height(bottom), heights)


def test_05(fixtures_dir):
    p=pd.read_csv(fixtures_dir/'proposals_sample.csv')
    gt=pd.read_csv(fixtures_dir/'tracks_gt.csv')
    assert proposals.proposal_recall(p,gt,.5)==pytest.approx(4/9)
    assert proposals.proposal_recall(p.iloc[:0],gt,.5)==0


@pytest.mark.data
@pytest.mark.skip(reason="TODO(Member 1, M3): real data: proposal recall at IoU 0.5 on 117093 eval10 is reported (no threshold asserted)")
def test_data_01(): ...


@pytest.fixture
def background_video(tmp_path, synthetic_frame):
    path = tmp_path / "background.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (640, 360))
    assert writer.isOpened()
    try:
        for color in [(20, 100, 180), (40, 120, 200), (60, 140, 220)]:
            writer.write(synthetic_frame(pitch_bgr=color))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_background_sampling_and_scale(background_video, scale):
    actual = segmentation.estimate_background(background_video, 20, 3, scale)
    assert actual.dtype == np.uint8
    assert actual.shape == (int(360 * scale), int(640 * scale), 3)
    assert (actual == np.array([40, 120, 200], dtype=np.uint8)).all()
    # Endpoint sampling with two frames has the same per-pixel median.
    np.testing.assert_array_equal(actual, segmentation.estimate_background(background_video, 2, 3, scale))


def test_background_refuses_incomplete_samples(background_video):
    with pytest.raises(ValueError, match="ended before"):
        segmentation.estimate_background(background_video, 3, 10, 0.5)


@pytest.mark.parametrize("scale", [0, -1, 1.1, float("nan"), True])
def test_invalid_background_scale(background_video, scale):
    with pytest.raises(ValueError, match="work_scale"):
        segmentation.estimate_background(background_video, 3, 3, scale)


def test_manual_full_resolution_polygon_maps_to_working_pixels(scratch_cfg, synthetic_frame):
    settings = scratch_cfg["segmentation"]
    settings["pitch_mask_method"] = "manual"
    settings["manual_polygons"] = {"999001": [[20, 20], [200, 20], [200, 100], [20, 100]]}
    mask = segmentation.pitch_mask(synthetic_frame(size=(320, 180)), scratch_cfg, "999001")
    rows, cols = np.where(mask)
    assert (cols.min(), rows.min(), cols.max(), rows.max()) == (10, 10, 100, 50)


def test_keypoint_fallback_requires_verified_pixel_space(scratch_cfg, synthetic_frame):
    scratch_cfg["segmentation"]["pitch_mask_method"] = "keypoints"
    with pytest.raises(NotImplementedError, match="verified calibration"):
        segmentation.pitch_mask(synthetic_frame(), scratch_cfg, "999001")


def test_empty_and_boundary_foreground(scratch_cfg, synthetic_frame):
    empty = np.zeros((30, 30), dtype=bool)
    assert not segmentation.clean_mask(empty, scratch_cfg).any()
    empty[:20, :8] = True
    clean = segmentation.clean_mask(empty, scratch_cfg)
    assert clean[0, 0]
    assert clean.shape == empty.shape
    assert not clean[-1, -1]
    frame = synthetic_frame()
    with pytest.raises(ValueError, match="matching resolutions"):
        segmentation.foreground_mask(frame, frame[:30], np.ones(frame.shape[:2], dtype=bool), scratch_cfg)


@pytest.mark.parametrize("bad", [np.zeros((10, 10)), np.zeros((10, 10, 3)), np.zeros((0, 10, 3), dtype=np.uint8)])
def test_invalid_image_inputs(scratch_cfg, bad):
    with pytest.raises(ValueError, match="BGR uint8"):
        segmentation.pitch_mask(bad, scratch_cfg, "999001")


@pytest.mark.parametrize("field,value", [("blur_ksize", 2), ("median_ksize", 2), ("open_ksize", 0), ("vertical_dilate", -1)])
def test_invalid_morphology_configuration(scratch_cfg, synthetic_frame, field, value):
    scratch_cfg["segmentation"][field] = value
    frame = synthetic_frame()
    with pytest.raises(ValueError):
        if field == "blur_ksize":
            segmentation.foreground_mask(frame, frame, np.ones(frame.shape[:2], dtype=bool), scratch_cfg)
        else:
            segmentation.clean_mask(np.zeros(frame.shape[:2], dtype=bool), scratch_cfg)


def test_save_mask_full_resolution_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(config, "load_config", lambda: {"project": {"paths": {"outputs_root": "outputs"}}})
    mask = np.zeros((10, 20), dtype=bool)
    mask[:5, :10] = True
    path = segmentation.save_pitch_mask(mask, "999001", (40, 20))
    assert path == tmp_path / "outputs/masks/999001_pitch.png"
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image.dtype == np.uint8
    assert image.shape == (20, 40)
    assert set(np.unique(image)) == {0, 255}
    assert image[0, 0] == 255 and image[-1, -1] == 0
    with pytest.raises(ValueError, match="match_id"):
        segmentation.save_pitch_mask(mask, "../outside", (40, 20))


def test_size_model_invalid_training_data_and_provenance():
    train = pd.DataFrame({"y": [100, 200], "h": [40, 50]})
    train.attrs["split"] = "val"
    with pytest.raises(PermissionError, match="train GT"):
        proposals.fit_size_model(train)
    with pytest.raises(ValueError, match="at least two"):
        proposals.fit_size_model(pd.DataFrame({"y": [100], "h": [0]}))
