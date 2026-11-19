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


# ---- Acceptance list for contracts.py (Member 3, M0) ----------------------------------------------------------
@pytest.mark.skip(reason="TODO(Member 3, M0): read_table + validate_* accept every committed fixture")
def test_validators_accept_fixtures(): ...


@pytest.mark.skip(reason="TODO(Member 3, M0): wrong column order, NaN, negative w/h, score > 1, duplicate (frame, track_id) are rejected")
def test_validators_reject_bad_tables(): ...


@pytest.mark.skip(reason="TODO(Member 3, M0): validate_metrics enforces value domains: bounded keys in [0, 1], MOTA may be negative (e.g. -0.5 accepted, 1.2 rejected), fp_per_frame/fps >= 0 and may exceed 1, idsw and counts are non-negative integers")
def test_metric_value_domains(): ...


@pytest.mark.skip(reason="TODO(Member 3, M0): detections outside the metadata frameset are rejected")
def test_detections_must_lie_in_frameset(): ...


@pytest.mark.skip(reason="TODO(Member 3, M0): write_table -> read_table round trip is lossless at contract precision")
def test_table_round_trip(): ...


@pytest.mark.skip(reason="TODO(Member 3, M0): unknown detector/tracker IDs are rejected by pipeline_id")
def test_identifier_vocabulary(): ...
