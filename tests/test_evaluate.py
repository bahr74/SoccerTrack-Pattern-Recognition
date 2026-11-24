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


def _many_gt(n_frames=200, players=22, size=(4096, 1080), seed=1):
    """A large synthetic GT Track table (players walking slowly), for statistical checks."""
    import numpy as np

    rng = np.random.default_rng(seed)
    start = np.column_stack([rng.uniform(0, size[0] - 40, players), rng.uniform(0, size[1] - 70, players)])
    rows = []
    for f in range(n_frames):
        pos = np.clip(start + f * 0.5, 0, [size[0] - 30, size[1] - 60])
        rows += [[f, i + 1, x, y, 20.0, 45.0, 1.0] for i, (x, y) in enumerate(pos)]
    return pd.DataFrame(rows, columns=C.TRACK_COLUMNS)


def test_03(gt, fixtures_dir):
    oracle = E.gt_as_detections(gt)
    meta = C.read_run_meta(fixtures_dir / "detections_sample.csv")
    C.validate_detections(oracle, dict(meta, method="gt_oracle"))
    assert len(oracle) == len(gt) and (oracle.score == 1.0).all()

    big = _many_gt()
    noisy = E.noisy_gt_detections(big, drop_rate=0.10, jitter_px=2.0, seed=42, image_size=(4096, 1080))
    C.validate_detections(noisy)
    C.validate_detections(noisy, dict(meta, method="gt_noisy", image_width=4096, image_height=1080,
                                      frames={"start": 0, "stop": 200, "stride": 1}))
    kept = len(noisy) / len(big)
    assert 0.88 <= kept <= 0.92                                   # about 10 % dropped
    assert noisy.score.between(0.3, 1.0).all()
    same = E.noisy_gt_detections(big, 0.10, 2.0, 42, (4096, 1080))
    pd.testing.assert_frame_equal(noisy, same)                    # fixed seed -> identical on every machine
    # Without drops, the noise is centred on the truth: the average box barely moves.
    centred = E.noisy_gt_detections(big, 0.0, 2.0, 7, (4096, 1080))
    assert len(centred) == len(big)
    assert abs(centred.x.mean() - big.x.mean()) < 0.1 and abs(centred.w.mean() - 20.0) < 0.1


def test_noisy_boxes_are_clipped_to_the_image(fixtures_dir):
    edge = pd.DataFrame([[0, 1, 0.0, 0.0, 20.0, 45.0, 1.0], [0, 2, 620.0, 315.0, 20.0, 45.0, 1.0]],
                        columns=C.TRACK_COLUMNS)
    for seed in range(20):
        noisy = E.noisy_gt_detections(edge, 0.0, 5.0, seed, image_size=(640, 360))
        C.validate_detections(noisy, {"image_width": 640, "image_height": 360,
                                      "frames": {"start": 0, "stop": 1, "stride": 1}, "score_floor": 0.05})


def test_write_oracle_detections(gt, cfg, tmp_path):
    paths = E.write_oracle_detections(gt, "val", "999001", "fixture3", range(0, 3), (640, 360), 25.0, cfg,
                                      outputs_root=tmp_path)
    assert paths["gt_oracle"] == tmp_path / "detections/gt_oracle/val/999001__fixture3.csv"
    for detector, path in paths.items():
        meta = C.read_run_meta(path)
        assert meta["method"] == detector
        C.validate_detections(C.read_table(path, "detections"), meta)
    # The oracle evaluated with E3 is perfect.
    m = E.evaluate_detections(paths["gt_oracle"], gt, cfg)["metrics"]
    assert m["precision"] == m["recall"] == m["ap50"] == 1.0


def _same_text(a, b):
    """Byte-for-byte comparison that ignores Windows CRLF checkouts (git core.autocrlf) of the committed files."""
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


TRACKEVAL_FILES = ("gt/999001__fixture3/gt/gt.txt", "gt/999001__fixture3/seqinfo.ini",
                   "trackers/yolov8n_coco+bytetrack/data/999001__fixture3.txt")


def test_04(fixtures_dir, gt, tmp_path):
    ws = E.export_trackeval(gt, fixtures_dir / "tracks_pred.csv", "E4__fixture", workspace_root=tmp_path)
    assert ws == tmp_path / "E4__fixture"
    for rel in TRACKEVAL_FILES:
        assert (ws / rel).read_bytes() == (ws / rel).read_bytes().replace(b"\r\n", b"\n")  # we write LF only
        assert _same_text(ws / rel, fixtures_dir / "trackeval_example" / rel), rel
    first = (ws / TRACKEVAL_FILES[0]).read_text(encoding="utf-8").splitlines()[0]
    assert first == "1,1,100.00,150.00,20.00,45.00,1,1,1"      # frame 0 -> 1; GT flags -1 -> 1


def _gt_as_tracker_output(gt, fixtures_dir, tmp_path):
    """The GT itself, written as if a tracker had produced it (pipeline gt_oracle+iou_hungarian)."""
    path = C.write_table(gt, tmp_path / "gt_tracks.csv", "tracks")
    meta = dict(C.read_run_meta(fixtures_dir / "tracks_pred.csv"), method="gt_oracle", tracker="iou_hungarian")
    C.write_run_meta(meta, path)
    return path


def test_05(fixtures_dir, gt, cfg, tmp_path):
    pytest.importorskip("trackeval")
    doc = E.evaluate_tracking(_gt_as_tracker_output(gt, fixtures_dir, tmp_path), gt, cfg,
                              workspace_root=tmp_path / "te")
    m = doc["metrics"]
    assert m["hota"] == m["idf1"] == m["mota"] == m["deta"] == m["assa"] == 1.0 and m["idsw"] == 0
    assert doc["run_id"] == "E4__gt_oracle+iou_hungarian__val__999001__fixture3"


def test_06(fixtures_dir, gt, cfg, tmp_path):
    pytest.importorskip("trackeval")
    doc = E.evaluate_tracking(fixtures_dir / "tracks_pred.csv", gt, cfg, workspace_root=tmp_path,
                              ground_truth="tests/fixtures/tracks_gt.csv")
    C.validate_metrics(doc)
    m, c = doc["metrics"], doc["counts"]
    assert (m["mota"], m["idf1"], m["idsw"]) == (0.6667, 0.75, 1)
    # Hand-computed in SHARED_CONTRACTS.md §15: FN 2, FP 0, IDSW 1 of 9 GT boxes; IDTP 6, IDFP 1, IDFN 3.
    assert (c["fn"], c["fp"], c["idtp"], c["idfp"], c["idfn"]) == (2, 0, 6, 1, 3)
    assert (c["gt_ids"], c["pred_ids"], c["gt"], c["pred"]) == (3, 4, 9, 7)
    assert 0 < m["hota"] < 1 and m["hota"] == pytest.approx((m["deta"] * m["assa"]) ** 0.5, abs=0.02)


def test_tracking_rejects_strided_framesets(fixtures_dir, gt, tmp_path):
    path = C.write_table(C.read_table(fixtures_dir / "tracks_pred.csv", "tracks").query("frame != 1"),
                         tmp_path / "t.csv", "tracks")
    C.write_run_meta(dict(C.read_run_meta(fixtures_dir / "tracks_pred.csv"),
                          frames={"start": 0, "stop": 3, "stride": 2}), path)
    with pytest.raises(C.ContractError, match="stride 1"):
        E.export_trackeval(gt, path, "x", workspace_root=tmp_path)


def _with_match(doc, match, **counts):
    """A copy of a Metrics JSON as if it came from another match, with some counts changed."""
    import copy

    out = copy.deepcopy(doc)
    out["match"] = match
    out["run_id"] = f"{doc['eval_id']}__{doc['method']}__{doc['split']}__{match}__{doc['frameset']}"
    out["counts"].update(counts)
    return out


def test_07(fixtures_dir, gt, cfg):
    e3 = E.evaluate_detections(fixtures_dir / "detections_sample.csv", gt, cfg)
    big = _with_match(e3, "999002", frames=100, gt=900, pred=500, tp=450, fp=50, fn=450)
    pooled = E.pool_metrics([e3, big])
    C.validate_metrics(pooled)
    assert pooled["run_id"] == "E3__yolov8n_coco__val__pooled__fixture3"
    assert pooled["counts"]["tp"] == 457 and pooled["counts"]["gt"] == 909
    m = pooled["metrics"]
    # Pooled = recomputed from summed counts (457 / 508), NOT the mean of 0.875 and 0.9.
    assert m["precision"] == round(457 / 508, 4) and m["recall"] == round(457 / 909, 4)
    assert m["precision"] != round((0.875 + 0.9) / 2, 4)
    assert m["fp_per_frame"] == round(51 / 103, 4)
    assert m["ap50"] is None                                       # AP needs pooled detections, not counts
    with pytest.raises(ValueError, match="once"):
        E.pool_metrics([e3, e3])
    with pytest.raises(ValueError, match="method"):
        E.pool_metrics([e3, dict(_with_match(e3, "999003"), method="gt_oracle",
                                 run_id="E3__gt_oracle__val__999003__fixture3")])


def test_pooled_ap_and_tracking(fixtures_dir, gt, cfg, tmp_path):
    pytest.importorskip("trackeval")
    det = fixtures_dir / "detections_sample.csv"
    # The same match twice is the same ranked list repeated, so the pooled AP equals the single-match AP.
    ap = E.pooled_detection_ap([(det, gt), (det, gt)], cfg)
    assert round(ap["ap50"], 4) == 0.8667
    e4 = E.evaluate_tracking(fixtures_dir / "tracks_pred.csv", gt, cfg, workspace_root=tmp_path)
    pooled = E.pool_metrics([e4, _with_match(e4, "999002")])
    m = pooled["metrics"]
    assert (m["mota"], m["idf1"], m["idsw"]) == (0.6667, 0.75, 2) and m["hota"] == e4["metrics"]["hota"]


def test_compare_tables_and_figures(fixtures_dir, gt, cfg, tmp_path):
    pytest.importorskip("matplotlib")
    from soccertrack import compare

    metrics_dir = tmp_path / "metrics"
    for doc in (E.evaluate_patches(fixtures_dir / "patch_scores_sample.csv", cfg),
                E.evaluate_proposals(fixtures_dir / "proposals_sample.csv", gt, cfg),
                E.evaluate_detections(fixtures_dir / "detections_sample.csv", gt, cfg),
                E.evaluate_tracking(fixtures_dir / "tracks_pred.csv", gt, cfg, workspace_root=tmp_path / "te")):
        E.write_metrics(doc, metrics_dir)
    long_df = compare.collect_metrics(metrics_dir)
    assert set(long_df.eval_id) == {"E1", "E2", "E3", "E4"}
    det = compare.detection_table(long_df, "val")
    assert det.loc[0, "approach"] == "B (YOLO)" and det.loc[0, "AP50"] == 0.8667
    trk = compare.tracking_table(long_df, "val")
    assert (trk.loc[0, "detector"], trk.loc[0, "tracker"], trk.loc[0, "IDSW"]) == ("yolov8n_coco", "bytetrack", 1)
    assert len(compare.classifier_table(long_df, "val")) == 2
    assert compare.detection_table(long_df, "test").empty
    refs = compare.reference_table()
    assert "no" in set(refs["comparable with our numbers?"])    # GS-HOTA is never compared with our HOTA
    written = compare.write_comparison(metrics_dir, "val", tmp_path / "tables", tmp_path / "figures",
                                       fixtures_dir / "eda_stats_sample.json")
    names = {p.name for p in written}
    assert {"detection_val.csv", "tracking_val.csv", "classifiers_val.csv", "references.csv",
            "dataset_statistics_val.csv", "pr_detection_val.png", "roc_classifiers_val.png"} <= names
    assert all(p.stat().st_size > 0 for p in written)


def test_runner_fixtures_command_and_blocked_commands(capsys):
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"
    spec = importlib.util.spec_from_file_location("run_evaluation", script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.frameset_range("eval10", 25, {"evaluation": {"framesets": {"eval10": {"start": 0, "stop": None,
                                                                                         "stride": 10}}}}) == range(0, 25, 10)
    with pytest.raises(SystemExit):
        runner.main(["detection"])                                # --detector is required
    with pytest.raises(PermissionError):
        runner.matches(runner.build_parser().parse_args(["detection", "--split", "test"]), {})
