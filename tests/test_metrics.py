"""Tests owned by Member 3. Each skip is an acceptance criterion: replace it with a real test.

Fixtures: tests/conftest.py and tests/fixtures/ (docs/team/SHARED_CONTRACTS.md, Section 15).
"""

from __future__ import annotations

import pytest


def test_01():
    np = pytest.importorskip("numpy")
    from soccertrack.metrics import iou_matrix

    a = np.array([[500, 140, 18, 40]])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[0, 0, 10, 10]]))[0, 0] == 0.0
    # Overlap 8 x 30 = 240; union 720 + 720 - 240 = 1200; IoU = 0.2.
    assert iou_matrix(a, np.array([[510, 150, 18, 40]]))[0, 0] == pytest.approx(0.2)
    # Touching edges share no area.
    assert iou_matrix(a, np.array([[518, 140, 18, 40]]))[0, 0] == 0.0
    # Shape [N, M] and empty inputs.
    m = iou_matrix(np.zeros((3, 4)) + [0, 0, 5, 5], np.zeros((2, 4)) + [0, 0, 5, 5])
    assert m.shape == (3, 2)
    assert iou_matrix(np.zeros((0, 4)), a).shape == (0, 1)


BOX = ["x", "y", "w", "h"]


def _fixture_matching(fixtures_dir, threshold, iou=0.5):
    """Match the fixture detections (score >= threshold) to the GT frame by frame; pool the results."""
    from soccertrack import contracts as C
    from soccertrack.metrics import match_frame

    det = C.read_table(fixtures_dir / "detections_sample.csv", "detections")
    gt = C.read_table(fixtures_dir / "tracks_gt.csv", "tracks")
    tp = fp = fn = 0
    flags, scores = [], []
    for frame in range(3):
        d = det[(det.frame == frame) & (det.score >= threshold)]
        r = match_frame(gt[gt.frame == frame][BOX].to_numpy(), d[BOX].to_numpy(), d.score.to_numpy(), iou)
        tp += int(r["tp_flags"].sum()); fp += int((~r["tp_flags"]).sum()); fn += r["fn"]
        flags += list(r["tp_flags"]); scores += list(d.score)
    return tp, fp, fn, flags, scores


def test_02(fixtures_dir):
    from soccertrack.metrics import precision_recall_f1

    tp, fp, fn, _, _ = _fixture_matching(fixtures_dir, threshold=0.5)
    assert (tp, fp, fn) == (7, 1, 2)
    p, r, f1 = precision_recall_f1(tp, fp, fn)
    assert (round(p, 4), round(r, 4), round(f1, 4)) == (0.875, 0.7778, 0.8235)
    assert precision_recall_f1(0, 0, 0) == (0.0, 0.0, 0.0)  # 0/0 is defined as 0


def test_match_frame_is_greedy_by_score_and_one_to_one():
    np = pytest.importorskip("numpy")
    from soccertrack.metrics import match_frame

    gt = np.array([[0, 0, 20, 40]])
    det = np.array([[1, 0, 20, 40], [0, 0, 20, 40]])            # the second box is a perfect match...
    r = match_frame(gt, det, np.array([0.6, 0.9]), 0.5)          # ...and has the higher score, so it wins
    assert r["tp_flags"].tolist() == [False, True] and r["matched_gt"].tolist() == [-1, 0] and r["fn"] == 0
    r = match_frame(gt, det, np.array([0.9, 0.6]), 0.5)          # now the first one picks first
    assert r["tp_flags"].tolist() == [True, False]
    assert match_frame(np.zeros((0, 4)), det, np.array([0.9, 0.6]), 0.5)["tp_flags"].sum() == 0
    assert match_frame(gt, np.zeros((0, 4)), np.zeros(0), 0.5)["fn"] == 1


def test_03(fixtures_dir):
    from soccertrack.metrics import average_precision

    tp, fp, fn, flags, scores = _fixture_matching(fixtures_dir, threshold=0.0)
    assert (tp, fp) == (8, 3)
    ap, curve = average_precision(flags, scores, num_gt=9)
    assert round(ap, 4) == 0.8667                     # 7/9 * 1.0 + 1/9 * 0.8
    assert len(curve["recall"]) == len(curve["precision"]) == 11
    perfect, _ = average_precision([True] * 9, list(range(9)), num_gt=9)
    assert perfect == pytest.approx(1.0)              # GT as prediction gives AP = 1
    assert average_precision([], [], num_gt=0)[0] is None


def test_04(fixtures_dir):
    from soccertrack import contracts as C
    from soccertrack.metrics import binary_classification_metrics

    t = C.read_table(fixtures_dir / "patch_scores_sample.csv", "patch_scores")
    r = binary_classification_metrics(t.y_true.to_numpy(), t.score.to_numpy(), 0.5)
    assert r["counts"] == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}
    assert r["metrics"]["sensitivity"] == 0.75 and r["metrics"]["specificity"] == 0.75
    assert r["metrics"]["roc_auc"] == pytest.approx(0.9375)
    assert set(r["curves"]) == {"roc", "pr"}
    one_class = binary_classification_metrics([1, 1], [0.2, 0.9], 0.5)
    assert one_class["metrics"]["roc_auc"] is None  # undefined, stored as null


def test_05():
    """Our all-point AP@0.5 agrees with pycocotools (101-point interpolation) within 0.01."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("pycocotools")
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from soccertrack.metrics import average_precision, match_frame

    rng = np.random.default_rng(0)
    images, annotations, results = [], [], []
    flags, scores, num_gt = [], [], 0
    for image_id in range(1, 61):
        images.append({"id": image_id, "width": 640, "height": 360})
        n = int(rng.integers(3, 9))
        gt = np.column_stack([rng.uniform(0, 580, n), rng.uniform(0, 300, n),
                              rng.uniform(15, 30, n), rng.uniform(35, 55, n)])
        num_gt += n
        for box in gt:
            annotations.append({"id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                                "bbox": box.tolist(), "area": float(box[2] * box[3]), "iscrowd": 0})
        keep = rng.random(n) < 0.85                                 # misses
        det = gt[keep] + rng.normal(0, 3, (int(keep.sum()), 4)) * [1, 1, 0.3, 0.3]
        fake = np.column_stack([rng.uniform(0, 580, 2), rng.uniform(0, 300, 2), np.full(2, 20), np.full(2, 45)])
        det = np.vstack([det, fake])                                # false positives
        det[:, 2:] = np.maximum(det[:, 2:], 2)
        s = rng.uniform(0.05, 1.0, len(det))
        r = match_frame(gt, det, s, 0.5)
        flags += list(r["tp_flags"]); scores += list(s)
        results += [{"image_id": image_id, "category_id": 1, "bbox": b.tolist(), "score": float(v)}
                    for b, v in zip(det, s)]
    ours, _ = average_precision(flags, scores, num_gt)

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "player"}]}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.iouThrs = np.array([0.5])
        e.params.maxDets = [1000]
        e.params.areaRng = [[0, 1e10]]
        e.params.areaRngLbl = ["all"]
        e.evaluate(); e.accumulate()
    precision = e.eval["precision"][0, :, 0, 0, 0]                  # [T, R, K, A, M] -> 101 recall points
    coco_ap = float(np.mean(precision[precision > -1]))
    assert abs(ours - coco_ap) <= 0.01
