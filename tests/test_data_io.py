"""Member 1 M1: synthetic MOT tables and temporary lossless video tests."""
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest

from soccertrack import config, contracts, data_io, split


@pytest.fixture
def validator(monkeypatch):
    """Mock Member 3 M0 only; test assertions check actual conversion results."""
    validate = Mock(side_effect=lambda df, image_size=None: df)
    monkeypatch.setattr(contracts, "validate_tracks", validate)
    return validate


@pytest.fixture
def tiny_video(tmp_path, synthetic_frame):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (640, 360))
    assert writer.isOpened(), "Installed OpenCV must support the FFV1 test codec"
    images = [synthetic_frame(pitch_bgr=(20+i, 100+i, 180+i)) for i in range(6)]
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()
    return path, images


def test_01(fixtures_dir):
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt")
    assert raw.shape == (9, 10)
    assert list(raw.columns) == ["frame", "id", "x", "y", "w", "h", "c7", "c8", "c9", "c10"]
    assert raw.frame.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert raw.id.tolist() == [1, 2, 3] * 3
    assert all(raw[c].dtype == np.dtype("int64") for c in ("frame", "id"))
    assert all(raw[c].dtype == np.dtype("float64") for c in ("x", "y", "w", "h"))
    assert (raw.loc[:, ["c7", "c8", "c9", "c10"]] == -1).all().all()


def test_02(fixtures_dir, validator):
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt")
    original = raw.copy(deep=True)
    tracks, issues = data_io.mot_to_tracks(raw.iloc[::-1], (640, 360), 3)
    pd.testing.assert_frame_equal(tracks, pd.read_csv(fixtures_dir / "tracks_gt.csv"))
    pd.testing.assert_frame_equal(raw, original)
    assert issues == dict(degenerate_boxes=0, clipped_boxes=0, duplicate_rows=0, frames_not_expected=0)
    validator.assert_called_once()
    assert validator.call_args.args[0] is tracks
    assert validator.call_args.kwargs == {"image_size": (640, 360)}


def test_03(fixtures_dir, validator):
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_dirty.txt")
    tracks, issues = data_io.mot_to_tracks(raw, (640, 360), 3)
    assert len(tracks) == 11
    assert issues == dict(degenerate_boxes=1, clipped_boxes=1, duplicate_rows=1, frames_not_expected=1)
    assert tracks.groupby("frame").size().tolist() == [3, 3, 3, 2]
    clipped = tracks.loc[(tracks.frame == 2) & (tracks.track_id == 2)].iloc[0]
    assert (clipped.x, clipped.y, clipped.w, clipped.h) == (630, 160, 10, 45)
    assert tuple(tracks.columns) == contracts.TRACK_COLUMNS
    assert not tracks.duplicated(["frame", "track_id"]).any()
    assert (tracks.score == 1).all()


def test_04(tiny_video):
    path, expected = tiny_video
    frames = list(data_io.iter_frames(path))
    assert [index for index, _ in frames] == list(range(6))
    for (_, image), original in zip(frames, expected, strict=True):
        assert image.dtype == np.uint8
        assert image.shape == (360, 640, 3)
        np.testing.assert_array_equal(image, original)  # includes BGR order
    selected = list(data_io.iter_frames(path, iter([4, 1, 4, 20])))
    assert [index for index, _ in selected] == [1, 4]
    for index, image in selected:
        np.testing.assert_array_equal(image, expected[index])
    assert list(data_io.iter_frames(path, [])) == []


def test_05(eda_sources, fixtures_dir, validator):
    """M2: real parser/cleaner/video decode, with only shared M0 boundaries mocked."""
    import json
    from datetime import datetime

    splits, sources = eda_sources
    stats = data_io.compute_eda_stats(splits, sources)
    fixture = json.loads((fixtures_dir / "eda_stats_sample.json").read_text(encoding="utf-8"))
    assert set(stats) == set(fixture)
    assert stats["totals"] == fixture["totals"]
    assert stats["paper_reference"] == fixture["paper_reference"]
    assert datetime.fromisoformat(stats["created_utc"]).utcoffset().total_seconds() == 0
    assert stats["created_utc"].endswith("Z")
    for match in fixture["matches"]:
        assert set(stats["matches"][match]) == set(fixture["matches"][match])
    assert stats["matches"]["999001"] == fixture["matches"]["999001"]
    dirty_expected = fixture["matches"]["999002"]
    # Linear interpolation over11 widths puts p05 halfway between10 and18.
    assert stats["matches"]["999002"] == dirty_expected
    # Persist/reload a synthetic stats artifact; reject NaN/Infinity in JSON.
    path = sources["synthetic_output"] / "stats.json"
    path.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == stats


@pytest.fixture
def eda_sources(tmp_path, fixtures_dir, synthetic_frame, monkeypatch):
    mapping = {}
    for match, file, frames in (("999001", "mot_raw_sample.txt", 3), ("999002", "mot_raw_dirty.txt", 4)):
        clip = tmp_path / f"{match}.avi"
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (640, 360))
        assert writer.isOpened()
        try:
            for _ in range(frames):
                writer.write(synthetic_frame())
        finally:
            writer.release()
        mapping[match] = config.MatchPaths(match, fixtures_dir / file, clip, None)
    monkeypatch.setattr(config, "match_paths", lambda match, sources: mapping[match])
    monkeypatch.setattr(config, "load_config", lambda: {"project": {
        "dataset": {"expected_boxes_per_frame": 3},
        "split": {"train": ["999002"], "val": ["999001"], "test": ["999099"]},
    }})
    return {"train": ["999002"], "val": ["999001"], "test": []}, {
        "matches": {key: {} for key in mapping}, "synthetic_output": tmp_path,
    }


def test_eda_refuses_test_before_any_path_access(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not load config"))
    resolver = Mock(side_effect=AssertionError("must not resolve paths"))
    monkeypatch.setattr(config, "load_config", loader)
    monkeypatch.setattr(config, "match_paths", resolver)
    with pytest.raises(PermissionError, match="held-out"):
        data_io.compute_eda_stats({"test": ["128057"]}, {})
    loader.assert_not_called()
    resolver.assert_not_called()


def test_eda_rejects_split_overlap_and_unknown_split(monkeypatch):
    monkeypatch.setattr(config, "load_config", Mock(side_effect=AssertionError("no access")))
    with pytest.raises(ValueError, match="repeated"):
        data_io.compute_eda_stats({"train": ["999001"], "val": ["999001"]}, {})
    with pytest.raises(ValueError, match="EDA splits"):
        data_io.compute_eda_stats({"validation": []}, {})


def test_eda_unavailable_subset_and_missing_configured_files(eda_sources, tmp_path, monkeypatch, validator):
    splits, sources = eda_sources
    sources["matches"].pop("999002")
    stats = data_io.compute_eda_stats(splits, sources)
    assert list(stats["matches"]) == ["999001"]
    assert stats["totals"] == {"matches": 1, "frames": 3, "boxes": 9}
    monkeypatch.setattr(config, "match_paths", lambda match, sources: config.MatchPaths(match, tmp_path / "missing.txt", tmp_path / "missing.avi", None))
    with pytest.raises(FileNotFoundError, match="Configured val match 999001"):
        data_io.compute_eda_stats(splits, sources)


def test_eda_coverage_counts_empty_and_missing_frames(eda_sources, tmp_path, monkeypatch, validator):
    splits, sources = eda_sources
    raw = pd.DataFrame([
        [0, 1, 10, 10, 20, 40, -1, -1, -1, -1],
        [2, 1, 10, 10, 1, 40, -1, -1, -1, -1],
    ])
    mot = tmp_path / "gaps.txt"
    raw.to_csv(mot, header=False, index=False)
    original = config.match_paths("999001", sources)
    monkeypatch.setattr(config, "match_paths", lambda match, sources: config.MatchPaths(match, mot, original.clip, None))
    stats = data_io.compute_eda_stats({"val": ["999001"]}, sources)
    match = stats["matches"]["999001"]
    assert match["frames_annotated"] == 2
    assert match["frames_decoded"] == 3
    assert match["boxes"] == 1
    assert match["boxes_per_frame"] == {"min": 0, "max": 1, "mean": 0.5}
    assert match["issues"] == dict(duplicate_rows=0, clipped_boxes=0, degenerate_boxes=1, frames_not_expected=3)


def test_eda_rejects_empty_annotations(eda_sources, tmp_path, monkeypatch, validator):
    _, sources = eda_sources
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    original = config.match_paths("999001", sources)
    monkeypatch.setattr(config, "match_paths", lambda match, sources: config.MatchPaths(match, empty, original.clip, None))
    with pytest.raises(ValueError, match="no annotations"):
        data_io.compute_eda_stats({"val": ["999001"]}, sources)


def test_eda_rejects_mislabeled_heldout_and_train_matches(eda_sources, monkeypatch):
    _, sources = eda_sources
    resolver = Mock(side_effect=AssertionError("must not resolve paths"))
    monkeypatch.setattr(config, "match_paths", resolver)
    with pytest.raises(PermissionError, match="held-out match"):
        data_io.compute_eda_stats({"val": ["999099"]}, sources)
    with pytest.raises(ValueError, match="belongs to train"):
        data_io.compute_eda_stats({"val": ["999002"]}, sources)
    resolver.assert_not_called()


@pytest.mark.data
def test_data_01(capsys):
    """Opt-in local train/val check; never open held-out test files."""
    sources_path = Path(__file__).resolve().parents[1] / "configs/data_sources.yaml"
    if not sources_path.is_file():
        pytest.skip("Real train/val check requires configured data_sources.yaml and shared M0")
    cfg = config.load_config()
    sources = config.load_data_sources(sources_path)
    with capsys.disabled():
        _check_configured_train_val(cfg, sources)


def _check_configured_train_val(cfg, sources):
    """Shared assertion path for the opt-in check and its synthetic regressions."""
    checked = []
    for name in ("train", "val"):
        for match in split.matches_for(name, cfg):
            if match not in sources["matches"]:
                continue
            paths = config.match_paths(match, sources)
            missing = [str(path) for path in (paths.clip, paths.mot) if not path.is_file()]
            assert not missing, f"Configured {name} match {match}: missing files: {missing}"
            raw = data_io.read_mot_raw(paths.mot)
            assert len(raw) > 0
            count = data_io.count_decoded_frames(paths.clip)
            assert count == int(raw.frame.max()) + 1
            assert raw.frame.nunique() == count
            checked.append(match)
            print(f"Checked {name} match {match}: {count} decoded/annotated frames")
    assert checked, "No configured train/val match has both local labels and video"
    return checked


@pytest.mark.parametrize("text", [
    "0,1,1,2,3,4,-1,-1,-1\n",
    "0,1,1,2,3,4,-1,-1,-1,-1,99\n",
    "0,1,1,2,3,4,-1,-1,-1,-1\n1,1,1,2,3,4,-1,-1,-1\n",
    "0.5,1,1,2,3,4,-1,-1,-1,-1\n",
    "0,1.5,1,2,3,4,-1,-1,-1,-1\n",
    "-1,1,1,2,3,4,-1,-1,-1,-1\n",
    "0,0,1,2,3,4,-1,-1,-1,-1\n",
    "0,1,nan,2,3,4,-1,-1,-1,-1\n",
    "0,1,inf,2,3,4,-1,-1,-1,-1\n",
    "0,1,word,2,3,4,-1,-1,-1,-1\n",
])
def test_reject_malformed_raw(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        data_io.read_mot_raw(path)


def test_empty_raw_and_tracks(tmp_path, validator):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    raw = data_io.read_mot_raw(path)
    tracks, issues = data_io.mot_to_tracks(raw, (640, 360))
    assert raw.shape == (0, 10)
    assert tracks.shape == (0, 7)
    assert tracks.frame.dtype == np.dtype("int64")
    assert set(issues.values()) == {0}


def test_clipping_degeneracy_and_missing_frames(fixtures_dir, validator):
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt").iloc[[0, 6]].copy()
    raw.loc[raw.frame == 0, ["x", "y", "w", "h"]] = [-10, -5, 30, 20]
    raw.loc[raw.frame == 2, "x"] = 700
    tracks, issues = data_io.mot_to_tracks(raw, (640, 360), 1)
    assert tracks.loc[0, ["x", "y", "w", "h"]].tolist() == [0, 0, 20, 15]
    assert issues == dict(duplicate_rows=0, degenerate_boxes=1, clipped_boxes=2, frames_not_expected=2)


def test_conflicting_identity_rejected(fixtures_dir, validator):
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt")
    conflict = raw.iloc[[0]].assign(x=101.0)
    with pytest.raises(ValueError, match="Conflicting"):
        data_io.mot_to_tracks(pd.concat([raw, conflict]), (640, 360))
    validator.assert_not_called()


def test_missing_validator_is_not_silently_bypassed(fixtures_dir, monkeypatch):
    def unavailable(*args, **kwargs):
        raise NotImplementedError("shared validator unavailable")
    monkeypatch.setattr(contracts, "validate_tracks", unavailable)
    with pytest.raises(NotImplementedError, match="shared validator"):
        data_io.mot_to_tracks(data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt"), (640, 360))


def test_load_gt_uses_official_paths(fixtures_dir, tiny_video, validator, monkeypatch):
    sources = {"synthetic": True}
    paths = config.MatchPaths("999001", fixtures_dir / "mot_raw_sample.txt", tiny_video[0], None)
    resolver = Mock(return_value=paths)
    monkeypatch.setattr(config, "match_paths", resolver)
    actual = data_io.load_gt_tracks("999001", sources)
    resolver.assert_called_once_with("999001", sources)
    pd.testing.assert_frame_equal(actual, pd.read_csv(fixtures_dir / "tracks_gt.csv"))


def test_video_metadata_and_count(tiny_video):
    path, _ = tiny_video
    assert data_io.video_info(path) == dict(width=640, height=360, fps=25.0, frame_count_header=6)
    assert data_io.count_decoded_frames(path) == 6


@pytest.mark.parametrize("ids", [[-1], [0.5], [True], ["1"]])
def test_invalid_frame_ids(tiny_video, ids):
    with pytest.raises(ValueError, match="frame_ids"):
        list(data_io.iter_frames(tiny_video[0], ids))


@pytest.mark.parametrize("operation", [data_io.video_info, data_io.count_decoded_frames, lambda p: list(data_io.iter_frames(p))])
def test_missing_video(tmp_path, operation):
    with pytest.raises(FileNotFoundError):
        operation(tmp_path / "absent.avi")


def test_grab_retrieve_and_release(tmp_path, synthetic_frame, monkeypatch):
    path = tmp_path / "fake.avi"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = True
    capture.grab.side_effect = [True] * 5 + [False]
    capture.retrieve.return_value = (True, synthetic_frame())
    capture.get.return_value = 999
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    assert [i for i, _ in data_io.iter_frames(path, [1, 4])] == [1, 4]
    assert capture.retrieve.call_count == 2
    capture.set.assert_not_called()
    capture.release.assert_called_once()
    capture.reset_mock()
    capture.grab.side_effect = [True] * 3 + [False]
    assert data_io.count_decoded_frames(path) == 3
    capture.get.assert_not_called()
    capture.release.assert_called_once()
    capture.reset_mock()
    capture.grab.side_effect = [True]
    iterator = data_io.iter_frames(path)
    next(iterator)
    iterator.close()
    capture.release.assert_called_once()


def test_capture_errors_release(tmp_path, monkeypatch):
    path = tmp_path / "fake.avi"
    path.touch()
    capture = Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    with pytest.raises(OSError, match="Cannot open"):
        data_io.video_info(path)
    capture.release.assert_called_once()
    capture.reset_mock()
    capture.isOpened.return_value = True
    capture.grab.return_value = True
    capture.retrieve.return_value = (False, None)
    with pytest.raises(OSError, match="decode frame 0"):
        list(data_io.iter_frames(path))
    capture.release.assert_called_once()


@pytest.mark.parametrize("column", ["frame", "id"])
@pytest.mark.parametrize("value", [2**53 + 1, 2**63 - 1])
def test_integer_precision_preserved(tmp_path, column, value, validator):
    row = ["0", "1", "100", "150", "20", "45", "-1", "-1", "-1", "-1"]
    row[0 if column == "frame" else 1] = str(value)
    path = tmp_path / "precise.txt"
    path.write_text(",".join(row) + "\n", encoding="utf-8")
    raw = data_io.read_mot_raw(path)
    assert raw[column].iloc[0] == value
    tracks, _ = data_io.mot_to_tracks(raw, (640, 360), 1)
    assert tracks["frame" if column == "frame" else "track_id"].iloc[0] == value


@pytest.mark.parametrize("value", ["1.0000000000000001", "9007199254740993.1", str(2**63), "NaN", "Infinity"])
@pytest.mark.parametrize("column", [0, 1])
def test_fractional_and_overflow_integer_tokens(tmp_path, column, value):
    row = ["0", "1", "100", "150", "20", "45", "-1", "-1", "-1", "-1"]
    row[column] = value
    path = tmp_path / "invalid-integer.txt"
    path.write_text(",".join(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        data_io.read_mot_raw(path)


@pytest.mark.parametrize("missing", ["mot", "clip"])
def test_real_data_check_rejects_missing_configured_file(tmp_path, fixtures_dir, tiny_video, monkeypatch, missing):
    paths = config.MatchPaths("999001", fixtures_dir / "mot_raw_sample.txt", tiny_video[0], None)
    absent = tmp_path / "missing-file"
    paths = config.MatchPaths(paths.match_id, absent if missing == "mot" else paths.mot,
                              absent if missing == "clip" else paths.clip, None)
    cfg = {"split": {"train": [], "val": ["999001"], "test": ["999099"]}}
    sources = {"matches": {"999001": {}, "999099": {"never": "open test"}}}
    resolver = Mock(return_value=paths)
    monkeypatch.setattr(config, "match_paths", resolver)
    with pytest.raises(AssertionError, match="Configured val match 999001: missing files"):
        _check_configured_train_val(cfg, sources)
    resolver.assert_called_once_with("999001", sources)


def test_real_data_check_reports_only_enabled_local_train_val(tmp_path, tiny_video, fixtures_dir, monkeypatch, capsys):
    cfg = {"split": {"train": [], "val": ["999001"], "val_optional": ["999002"],
                     "use_val_optional": False, "test": ["999099"]}}
    sources = {"matches": {"999001": {}, "999002": {}, "999099": {}}}
    raw = data_io.read_mot_raw(fixtures_dir / "mot_raw_sample.txt")
    mot = tmp_path / "six-frames.txt"
    pd.concat([raw, raw.assign(frame=raw.frame + 3)]).to_csv(mot, header=False, index=False)
    paths = config.MatchPaths("999001", mot, tiny_video[0], None)
    resolver = Mock(return_value=paths)
    monkeypatch.setattr(config, "match_paths", resolver)
    assert _check_configured_train_val(cfg, sources) == ["999001"]
    assert "Checked val match 999001" in capsys.readouterr().out
    resolver.assert_called_once_with("999001", sources)
