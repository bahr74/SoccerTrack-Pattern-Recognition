"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


pd = pytest.importorskip("pandas")
from soccertrack import contracts as C  # noqa: E402
from soccertrack import evaluate as E  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return E.load_evaluation_config()


@pytest.fixture()
def gt(fixtures_dir):
    return C.read_table(fixtures_dir / "tracks_gt.csv", "tracks")


def test_01(fixtures_dir, gt, cfg, tmp_path):
    doc = E.evaluate_detections(fixtures_dir / "detections_sample.csv", gt, cfg,
                                ground_truth="tests/fixtures/tracks_gt.csv")
    C.validate_metrics(doc)
    assert doc["run_id"] == "E3__yolov8n_coco__val__999001__fixture3"
    assert doc["counts"] == {"frames": 3, "gt": 9, "pred": 8, "tp": 7, "fp": 1, "fn": 2}
    m = doc["metrics"]
    assert (m["precision"], m["recall"], m["f1"], m["ap50"]) == (0.875, 0.7778, 0.8235, 0.8667)
    assert m["fp_per_frame"] == 0.3333 and "fps" not in m       # the fixture metadata has no FPS
    assert 0 <= m["ap50_95"] <= m["ap50"]
    # Same values as the committed hand-written example (except ap50_95, which is illustrative there).
    expected = __import__("json").load(open(fixtures_dir / "metrics_sample.json", encoding="utf-8"))
    for key in ("precision", "recall", "f1", "ap50", "fp_per_frame"):
        assert m[key] == expected["metrics"][key]
    assert doc["breakdown"]["recall_by_height_px"] == expected["breakdown"]["recall_by_height_px"]
    assert doc["breakdown"]["at_iou_0.3"]["recall"] >= m["recall"]  # a looser IoU can only find more
    assert doc["inputs"]["prediction"] == "tests/fixtures/detections_sample.csv"
    path = E.write_metrics(doc, out_dir=tmp_path)
    assert path.name == "E3__yolov8n_coco__val__999001__fixture3.json"
    assert __import__("json").loads(path.read_text(encoding="utf-8")) == doc


def test_gt_as_prediction_is_perfect(gt, cfg, tmp_path, fixtures_dir):
    det = gt.drop(columns="track_id")
    path = C.write_table(det, tmp_path / "gt.csv", "detections")
    C.write_run_meta(C.read_run_meta(fixtures_dir / "detections_sample.csv"), path)
    m = E.evaluate_detections(path, gt, cfg)["metrics"]
    assert m["precision"] == m["recall"] == m["ap50"] == m["ap50_95"] == 1.0 and m["fp_per_frame"] == 0


def test_02(fixtures_dir, gt, cfg):
    e1 = E.evaluate_patches(fixtures_dir / "patch_scores_sample.csv", cfg)
    assert e1["run_id"] == "E1__scratch_hog_svm__val__pooled__patches"
    assert e1["counts"] == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}
    assert e1["metrics"]["roc_auc"] == 0.9375 and e1["metrics"]["sensitivity"] == 0.75
    assert e1["breakdown"]["accuracy_by_kind"]["partial"] == 0.0  # the one hard negative

    e2 = E.evaluate_proposals(fixtures_dir / "proposals_sample.csv", gt, cfg)
    assert e2["run_id"] == "E2__scratch_hog_svm__val__999001__fixture3"
    # 6 proposals: 4 overlap a GT box with IoU >= 0.5 and all four score >= 0.5; the 2 others score below.
    assert e2["counts"] == {"tp": 4, "fp": 0, "tn": 2, "fn": 0, "proposals": 6, "gt": 9,
                            "unscored": 0, "positives": 4}
    for doc in (e1, e2):
        C.validate_metrics(doc)


@pytest.mark.skip(reason="TODO(Member 3, M2): gt_as_detections and noisy_gt_detections produce valid Detection tables; noisy drop rate within tolerance")
def test_03(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): export_trackeval(tracks_gt.csv, tracks_pred.csv) reproduces tests/fixtures/trackeval_example byte for byte")
def test_04(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): GT evaluated against itself gives HOTA = IDF1 = MOTA = 1.0")
def test_05(): ...


@pytest.mark.skip(reason="TODO(Member 3, M3): tracks_pred.csv gives MOTA 0.6667, IDF1 0.7500, IDSW 1")
def test_06(): ...


@pytest.mark.skip(reason="TODO(Member 3, M4): pool_metrics recomputes P, R and F1 from summed counts")
def test_07(): ...
