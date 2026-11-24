"""Contract tests. Owner: Member 3.

Part 1 runs NOW (standard library only): the committed fixtures must match the schema constants.
Part 2 is the acceptance list for contracts.py (milestone M0); replace each skip with a real test.
"""

from __future__ import annotations

import csv
import json

import pytest

from soccertrack import contracts as C

CSV_FIXTURES = {
    "tracks_gt.csv": C.TRACK_COLUMNS,
    "tracks_pred.csv": C.TRACK_COLUMNS,
    "detections_sample.csv": C.DETECTION_COLUMNS,
    "proposals_sample.csv": C.PROPOSAL_COLUMNS,
    "patch_scores_sample.csv": C.PATCH_SCORE_COLUMNS,
    "pitch_tracks_sample.csv": C.PITCH_TRACK_COLUMNS,
}


@pytest.mark.parametrize("name,columns", sorted(CSV_FIXTURES.items()))
def test_fixture_header_matches_schema(fixtures_dir, name, columns):
    with open(fixtures_dir / name, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert tuple(header) == columns


def test_raw_mot_fixtures_have_ten_columns_and_zero_based_frames(fixtures_dir):
    for name in ("mot_raw_sample.txt", "mot_raw_dirty.txt"):
        rows = list(csv.reader(open(fixtures_dir / name, encoding="utf-8")))
        assert all(len(r) == C.RAW_MOT_NUM_COLUMNS for r in rows)
        assert min(int(r[0]) for r in rows) == 0
        assert all(r[6:] == ["-1", "-1", "-1", "-1"] for r in rows)


@pytest.mark.parametrize("name", ["detections_sample.meta.json", "tracks_pred.meta.json"])
def test_fixture_meta_has_required_keys(fixtures_dir, name):
    meta = json.load(open(fixtures_dir / name, encoding="utf-8"))
    assert set(C.RUN_META_REQUIRED_KEYS) <= set(meta)
    assert meta["kind"] in C.ARTIFACT_KINDS


def test_metrics_sample_has_required_keys(fixtures_dir):
    m = json.load(open(fixtures_dir / "metrics_sample.json", encoding="utf-8"))
    assert set(C.METRICS_REQUIRED_KEYS) <= set(m)
    assert C.EVAL_TASKS[m["eval_id"]] == m["task"]
    assert set(C.REQUIRED_METRIC_KEYS[m["task"]]) <= set(m["metrics"])
    assert set(C.REQUIRED_COUNT_KEYS[m["task"]]) <= set(m["counts"])


def test_trackeval_example_is_one_based(fixtures_dir):
    base = fixtures_dir / "trackeval_example"
    gt = list(csv.reader(open(base / "gt/999001__fixture3/gt/gt.txt", encoding="utf-8")))
    trk = list(csv.reader(open(base / "trackers/yolov8n_coco+bytetrack/data/999001__fixture3.txt", encoding="utf-8")))
    assert min(int(r[0]) for r in gt) == 1 and all(r[6:] == ["1", "1", "1"] for r in gt)
    assert all(len(r) == 10 and r[7:] == ["-1", "-1", "-1"] for r in trk)


# ---- Acceptance tests for contracts.py (Member 3, M0) ---------------------------------------------------------
pd = pytest.importorskip("pandas")

TABLE_FIXTURES = {
    "tracks_gt.csv": "tracks",
    "tracks_pred.csv": "tracks",
    "detections_sample.csv": "detections",
    "proposals_sample.csv": "proposals",
    "patch_scores_sample.csv": "patch_scores",
    "pitch_tracks_sample.csv": "pitch_tracks",
}


@pytest.fixture()
def gt(fixtures_dir):
    return C.read_table(fixtures_dir / "tracks_gt.csv", "tracks")


@pytest.fixture()
def det_meta(fixtures_dir):
    return C.read_run_meta(fixtures_dir / "detections_sample.csv")


@pytest.mark.parametrize("name,kind", sorted(TABLE_FIXTURES.items()))
def test_validators_accept_fixtures(fixtures_dir, name, kind):
    df = C.read_table(fixtures_dir / name, kind)
    assert tuple(df.columns) == C.TABLE_SCHEMAS[kind] and len(df) > 0


def test_validators_accept_fixture_metadata_and_metrics(fixtures_dir, det_meta):
    det = C.read_table(fixtures_dir / "detections_sample.csv", "detections")
    C.validate_detections(det, det_meta)
    C.validate_tracks(C.read_table(fixtures_dir / "tracks_gt.csv", "tracks"), image_size=(640, 360))
    assert C.read_run_meta(fixtures_dir / "tracks_pred.csv")["tracker"] == "bytetrack"
    C.validate_metrics(json.load(open(fixtures_dir / "metrics_sample.json", encoding="utf-8")))


def test_patch_set_validation(synthetic_patches):
    import numpy as np

    X, y = synthetic_patches(n_pos=2, n_neg=2)
    arrays = {"X": X, "y": y, "match": np.array(["999001"] * 4, dtype="<U6"), "frame": np.zeros(4, np.int32),
              "box": np.full((4, 4), 20, np.float32), "kind": np.array(["gt", "gt", "random", "partial"], dtype="<U8"),
              "meta_json": np.array(['{"schema_version": "1.0"}'])}
    C.validate_patch_set(arrays)
    with pytest.raises(C.ContractError, match="kind"):
        C.validate_patch_set(dict(arrays, kind=np.array(["gt", "gt", "random", "grass"], dtype="<U8")))
    with pytest.raises(C.ContractError, match="X must be"):
        C.validate_patch_set(dict(arrays, X=X[:, :32]))


@pytest.mark.parametrize("change,message", [
    (lambda d: d[["track_id", "frame", "x", "y", "w", "h", "score"]], "columns"),
    (lambda d: d.assign(x=d.x.where(d.index != 2)), "NaN"),
    (lambda d: d.assign(w=d.w.where(d.index != 0, -5.0)), "width and height"),
    (lambda d: d.assign(score=d.score.where(d.index != 0, 1.5)), r"\[0, 1\]"),
    (lambda d: pd.concat([d, d.iloc[[0]]]).sort_values(["frame", "track_id"]), "duplicate"),
    (lambda d: d.iloc[::-1], "sorted"),
    (lambda d: d.assign(track_id=d.track_id.where(d.index != 0, 0)), "track_id"),
])
def test_validators_reject_bad_tables(gt, change, message):
    with pytest.raises(C.ContractError, match=message):
        C.validate_tracks(change(gt).reset_index(drop=True))


def test_validators_reject_boxes_outside_image(gt):
    moved = gt.assign(x=gt.x.where(gt.index != 0, 630.0))
    C.validate_tracks(moved)  # image size unknown: cannot check the right border
    with pytest.raises(C.ContractError, match="outside"):
        C.validate_tracks(moved, image_size=(640, 360))


def test_detection_score_floor(fixtures_dir, det_meta):
    det = C.read_table(fixtures_dir / "detections_sample.csv", "detections")
    low = det.assign(score=det.score.where(det.index != 3, 0.01))
    with pytest.raises(C.ContractError, match="score_floor"):
        C.validate_detections(low, det_meta)


def _metrics(fixtures_dir, **changes):
    m = json.load(open(fixtures_dir / "metrics_sample.json", encoding="utf-8"))
    for key, value in changes.items():
        m["metrics"][key] = value
    return m


def _tracking_metrics(**values):
    run_id = "E4__gt_oracle+bytetrack__val__999001__fixture3"
    metrics = {"hota": 0.5, "deta": 0.5, "assa": 0.5, "loca": 0.9, "idf1": 0.7, "mota": 0.6, "idsw": 1}
    metrics.update(values)
    return {"schema_version": "1.0", "run_id": run_id, "eval_id": "E4", "task": "tracking",
            "method": "gt_oracle+bytetrack", "split": "val", "match": "999001", "frameset": "fixture3",
            "params": {}, "counts": {"frames": 3, "gt_ids": 3, "pred_ids": 3, "gt": 9, "pred": 9},
            "metrics": metrics, "inputs": {}, "provenance": {}}


def test_metric_value_domains(fixtures_dir):
    C.validate_metrics(_metrics(fixtures_dir, fp_per_frame=3.7))          # may exceed 1
    C.validate_metrics(_metrics(fixtures_dir, ap50_95=None))              # undefined value -> null
    C.validate_metrics(_tracking_metrics(mota=-0.5))                      # MOTA may be negative
    C.validate_metrics(_tracking_metrics(mota=1.0, idsw=0))
    for bad in (_metrics(fixtures_dir, precision=87.5),                   # percentage instead of fraction
                _metrics(fixtures_dir, recall=-0.1),
                _metrics(fixtures_dir, fp_per_frame=-1.0),
                _tracking_metrics(mota=1.2),
                _tracking_metrics(idsw=1.5),
                _tracking_metrics(idsw=-1),
                _tracking_metrics(hota=float("nan"))):
        with pytest.raises(C.ContractError):
            C.validate_metrics(bad)
    negative_count = _metrics(fixtures_dir)
    negative_count["counts"]["tp"] = -1
    with pytest.raises(C.ContractError, match="counts.tp"):
        C.validate_metrics(negative_count)
    missing = _metrics(fixtures_dir)
    del missing["metrics"]["ap50"]
    with pytest.raises(C.ContractError, match="ap50"):
        C.validate_metrics(missing)
    wrong_id = _metrics(fixtures_dir)
    wrong_id["run_id"] = "E3__something_else"
    with pytest.raises(C.ContractError, match="run_id"):
        C.validate_metrics(wrong_id)


def test_detections_must_lie_in_frameset(fixtures_dir, det_meta):
    det = C.read_table(fixtures_dir / "detections_sample.csv", "detections")
    outside = det.assign(frame=det.frame.where(det.index != 10, 3))      # fixture3 = frames 0, 1, 2
    with pytest.raises(C.ContractError, match="frameset"):
        C.validate_detections(outside, det_meta)
    strided = dict(det_meta, frames={"start": 0, "stop": 3, "stride": 2})  # frames 0 and 2 only
    with pytest.raises(C.ContractError, match=r"\[1\]"):
        C.validate_detections(det, strided)


@pytest.mark.parametrize("name,kind", sorted(TABLE_FIXTURES.items()))
def test_table_round_trip(fixtures_dir, tmp_path, name, kind):
    original = C.read_table(fixtures_dir / name, kind)
    out = C.write_table(original, tmp_path / "out" / name, kind)
    again = C.read_table(out, kind)
    pd.testing.assert_frame_equal(original.reset_index(drop=True), again)
    # Committed fixtures are already in the canonical format, so the bytes are identical too.
    assert out.read_bytes() == (fixtures_dir / name).read_bytes()


def test_write_table_sorts_and_rounds(tmp_path, gt):
    messy = gt.iloc[::-1].assign(x=gt.x + 0.004)
    C.write_table(messy, tmp_path / "t.csv", "tracks")
    lines = (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()
    assert lines[1] == "0,1,100.00,150.00,20.00,45.00,1.0000"


def test_run_meta_round_trip_and_rules(fixtures_dir, tmp_path, det_meta):
    C.write_run_meta(det_meta, tmp_path / "x.csv")
    assert C.read_run_meta(tmp_path / "x.csv") == det_meta
    for bad in (dict(det_meta, match=999001),                 # match must be a quoted string
                dict(det_meta, kind="tracks"),                # tracks need a tracker
                dict(det_meta, split="test"),                 # test artefacts must be frozen
                dict(det_meta, method="yolov9")):             # closed vocabulary
        with pytest.raises(C.ContractError):
            C.validate_run_meta(bad)


def test_identifier_vocabulary():
    assert C.pipeline_id("yolov8n_coco", "bytetrack") == "yolov8n_coco+bytetrack"
    assert C.split_pipeline_id("gt_oracle+iou_hungarian") == ("gt_oracle", "iou_hungarian")
    with pytest.raises(C.ContractError, match="detector"):
        C.pipeline_id("yolov9", "bytetrack")
    with pytest.raises(C.ContractError, match="tracker"):
        C.pipeline_id("yolov8n_coco", "sort")
