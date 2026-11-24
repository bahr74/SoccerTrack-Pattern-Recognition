"""Evaluation drivers E1-E4, oracle detection inputs and the TrackEval adapter.

Owner: Member 3. Config: configs/evaluation.yaml.
Reads only contract artefacts (Detection/Track/Proposal/Patch score tables + metadata) and writes Metrics JSON
(Contracts §8) to outputs/metrics/<run_id>.json. Never imports detector or tracker modules.
The SAME functions evaluate Pipeline A and Pipeline B (fair comparison).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from soccertrack import contracts, metrics as M

if TYPE_CHECKING:
    import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_BOX = ["x", "y", "w", "h"]


# ---- Shared helpers ----------------------------------------------------------------------------------------------
def load_evaluation_config() -> dict[str, Any]:
    """Configuration for the evaluation: {"evaluation": ..., "project": ...}.

    Uses config.load_config() (Member 4). Until that function exists, reads the two YAML files directly so the
    evaluation can be developed and tested on fixtures (Mode 1).
    """
    try:
        from soccertrack import config

        return config.load_config()
    except NotImplementedError:
        import yaml

        cfg = {}
        for name in ("evaluation", "project"):
            with open(REPO_ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
                cfg[name] = yaml.safe_load(f)
        return cfg


def _ev(cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept either the full config ({"evaluation": ...}) or just the evaluation section."""
    return cfg["evaluation"] if "evaluation" in cfg else cfg


def _relative(path: Path | str) -> str:
    """Repository-relative POSIX path (Contracts §1: never absolute paths inside artefacts)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def _frames(meta: dict[str, Any]) -> range:
    f = meta["frames"]
    return range(f["start"], f["stop"], f["stride"])


def _gt_in_frames(gt_tracks: "pd.DataFrame", frames: range) -> "pd.DataFrame":
    """GT restricted to the frames the producer actually processed (fair: same frames for A and B)."""
    import numpy as np

    f = gt_tracks["frame"].to_numpy()
    keep = (f >= frames.start) & (f < frames.stop) & ((f - frames.start) % frames.step == 0)
    return gt_tracks.loc[keep]


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def _provenance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Where a number came from: config hash, git commit, time, package versions, seed."""
    import subprocess
    from datetime import datetime, timezone
    from importlib import metadata

    try:
        from soccertrack import config

        sha = config.config_sha256()
    except NotImplementedError:  # Member 4 M0 not merged yet: flagged explicitly, never invented
        sha = "0" * 64
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "trackeval"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    out = {"config_sha256": sha, "git_commit": commit,
           "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "package_versions": versions, "seed": cfg.get("project", {}).get("seed", 42)}
    if sha == "0" * 64:
        out["note"] = "config.config_sha256 not implemented yet (Member 4 M0); placeholder hash"
    return out


def _document(eval_id: str, method: str, split: str, match: str, frameset: str, params: dict, counts: dict,
              values: dict, inputs: dict, cfg: dict, breakdown: dict | None = None,
              curves: dict | None = None) -> dict[str, Any]:
    """Assemble a Metrics JSON document (Contracts §8) and validate it."""
    doc = {
        "schema_version": contracts.SCHEMA_VERSION,
        "run_id": f"{eval_id}__{method}__{split}__{match}__{frameset}",
        "eval_id": eval_id, "task": contracts.EVAL_TASKS[eval_id], "method": method, "split": split,
        "match": match, "frameset": frameset, "params": params, "counts": counts,
        "metrics": {k: (_round(v) if isinstance(v, float) else v) for k, v in values.items()},
        "inputs": inputs, "provenance": _provenance(cfg),
    }
    if breakdown:
        doc["breakdown"] = breakdown
    if curves:
        doc["curves"] = curves
    return contracts.validate_metrics(doc)


def _classification_document(eval_id: str, y_true, scores, threshold: float, meta: dict, inputs: dict,
                             cfg: dict, extra_counts: dict | None = None, breakdown: dict | None = None,
                             match: str | None = None, frameset: str | None = None) -> dict[str, Any]:
    """E1 and E2 share the same binary-classification metrics."""
    points = _ev(cfg).get("classification", {}).get("curve_points", 200)
    result = M.binary_classification_metrics(y_true, scores, threshold, max_points=points)
    counts = dict(result["counts"], **(extra_counts or {}))
    return _document(eval_id, meta["method"], meta["split"], match or meta.get("match") or "pooled",
                     frameset or meta.get("frameset") or "patches",
                     {"score_threshold": threshold}, counts, result["metrics"], inputs, cfg,
                     breakdown=breakdown, curves=result["curves"])


# ---- Oracle inputs for tracker validation (used by Members 1 and 2) --------------------------------------------
def gt_as_detections(gt_tracks: "pd.DataFrame") -> "pd.DataFrame":
    """Detector `gt_oracle`: drop track_id, keep score 1.0.

    A "perfect detector": the real boxes, without identities. Feeding it to a tracker measures the tracker ALONE,
    because every tracking error must then come from the association step, not from detection mistakes.
    """
    det = gt_tracks.loc[:, list(contracts.DETECTION_COLUMNS)].copy()
    det["score"] = 1.0
    return det.sort_values(["frame"], kind="mergesort").reset_index(drop=True)


def noisy_gt_detections(gt_tracks: "pd.DataFrame", drop_rate: float, jitter_px: float, seed: int,
                        image_size: tuple[int, int] | None = None) -> "pd.DataFrame":
    """Detector `gt_noisy`: randomly drop boxes, jitter coordinates, assign scores in [0.3, 1.0]; clip to image.

    A "realistic but controlled" detector: some players are missed (drop_rate), boxes are slightly off
    (Gaussian noise with standard deviation jitter_px on x, y, w, h) and confidences vary. The fixed seed makes the
    result identical on every machine, so both trackers see exactly the same noisy input.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    keep = rng.random(len(gt_tracks)) >= drop_rate
    det = gt_tracks.loc[keep, list(contracts.DETECTION_COLUMNS)].copy().reset_index(drop=True)
    boxes = det[_BOX].to_numpy(dtype=float) + rng.normal(0.0, jitter_px, (len(det), 4))
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    x1, y1 = np.maximum(x1, 0.0), np.maximum(y1, 0.0)               # clip to the image
    if image_size is not None:
        x2, y2 = np.minimum(x2, image_size[0]), np.minimum(y2, image_size[1])
    # Keep every box valid (w, h > 1) even after noise and clipping.
    x2, y2 = np.maximum(x2, x1 + 2.0), np.maximum(y2, y1 + 2.0)
    if image_size is not None:
        x1, y1 = np.minimum(x1, x2 - 2.0), np.minimum(y1, y2 - 2.0)
    det[_BOX] = np.column_stack([x1, y1, x2 - x1, y2 - y1]).round(2)
    det["score"] = rng.uniform(0.3, 1.0, len(det)).round(4)
    return det.sort_values(["frame", "score"], ascending=[True, False], kind="mergesort").reset_index(drop=True)


def oracle_detection_path(detector: str, split: str, match: str, frameset: str, outputs_root: Path | None = None) -> Path:
    """outputs/detections/<detector>/<split>/<match>__<frameset>.csv (Contracts §2)."""
    root = Path(outputs_root) if outputs_root is not None else REPO_ROOT / "outputs"
    return root / "detections" / detector / split / f"{match}__{frameset}.csv"


def write_oracle_detections(gt_tracks: "pd.DataFrame", split: str, match: str, frameset: str, frames: range,
                            image_size: tuple[int, int], fps: float, cfg: dict[str, Any],
                            outputs_root: Path | None = None, frozen_test: bool = False) -> dict[str, Path]:
    """Write the `gt_oracle` and `gt_noisy` Detection tables + metadata for one clip and frameset.

    These are the shared inputs that Members 1 and 2 feed to their trackers for the oracle rows of E4.
    """
    import time
    from datetime import datetime, timezone

    ev = _ev(cfg)
    noise = ev["oracle"]["noisy_gt"]
    gt = _gt_in_frames(gt_tracks, frames)
    provenance = _provenance(cfg)
    written = {}
    for detector in ("gt_oracle", "gt_noisy"):
        start = time.perf_counter()
        if detector == "gt_oracle":
            det = gt_as_detections(gt)
        else:
            det = noisy_gt_detections(gt, noise["drop_rate"], noise["jitter_px"], noise["seed"], image_size)
        seconds = time.perf_counter() - start
        meta = {
            "schema_version": contracts.SCHEMA_VERSION, "kind": "detections", "method": detector, "tracker": None,
            "split": split, "match": match, "frameset": frameset,
            "frames": {"start": frames.start, "stop": frames.stop, "stride": frames.step},
            "image_width": int(image_size[0]), "image_height": int(image_size[1]), "fps": float(fps),
            "score_floor": float(ev["score_floor"]), "operating_threshold": 0.0 if detector == "gt_oracle" else 0.3,
            "frozen_test": frozen_test, "config_sha256": provenance["config_sha256"], "git_commit": None,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "runtime_seconds": round(seconds, 4), "frames_per_second": None, "device": "cpu",
            "notes": "oracle input" if detector == "gt_oracle" else
                     f"noisy oracle: drop {noise['drop_rate']}, jitter {noise['jitter_px']} px, seed {noise['seed']}",
        }
        contracts.validate_detections(det, meta)
        path = oracle_detection_path(detector, split, match, frameset, outputs_root)
        contracts.write_table(det, path, "detections")
        contracts.write_run_meta(meta, path)
        written[detector] = path
    return written


# ---- E1-E4 -------------------------------------------------------------------------------------------------------
def evaluate_patches(scores_csv: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """E1 from a Patch score table + metadata (operating_threshold from the metadata).

    Each row is one 64x32 patch with its true label (y_true) and the classifier score. score >= threshold means
    "player". Also reports accuracy per patch kind (gt / random / partial / proposal) to show which negatives are hard.
    """
    table = contracts.read_table(scores_csv, "patch_scores")
    meta = contracts.read_run_meta(scores_csv)
    threshold = float(meta["operating_threshold"])
    correct = (table["score"] >= threshold).astype(int) == table["y_true"]
    by_kind = {kind: round(float(correct[table["kind"] == kind].mean()), 4) for kind in sorted(table["kind"].unique())}
    return _classification_document("E1", table["y_true"], table["score"], threshold, meta,
                                     {"prediction": _relative(scores_csv)}, cfg,
                                     breakdown={"accuracy_by_kind": by_kind})


def evaluate_proposals(proposals_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any],
                       ground_truth: str | None = None) -> dict[str, Any]:
    """E2: label scored proposals by IoU >= proposal_eval.iou_positive with GT; classification metrics.

    Unlike E1 (balanced 1:3 patches), this uses EVERY candidate region the segmentation produced, so it shows the
    real imbalance between players and background. Unscored proposals (score = -1) are excluded and counted.
    """
    import numpy as np

    table = contracts.read_table(proposals_csv, "proposals")
    meta = contracts.read_run_meta(proposals_csv)
    contracts.validate_proposals(table, meta)
    iou_positive = float(_ev(cfg)["proposal_eval"]["iou_positive"])
    gt = _gt_in_frames(gt_tracks, _frames(meta))
    scored = table.loc[table["score"] != -1.0]
    labels = np.zeros(len(scored), dtype=int)
    for frame, rows in scored.groupby("frame"):
        gt_boxes = gt.loc[gt["frame"] == frame, _BOX].to_numpy()
        if len(gt_boxes):
            best = M.iou_matrix(rows[_BOX].to_numpy(), gt_boxes).max(axis=1)
            labels[scored.index.get_indexer(rows.index)] = best >= iou_positive
    threshold = float(meta["operating_threshold"])
    extra = {"proposals": int(len(scored)), "gt": int(len(gt)), "unscored": int(len(table) - len(scored)),
             "positives": int(labels.sum())}
    inputs = {"prediction": _relative(proposals_csv), "ground_truth": ground_truth or f"soccertrack:{meta['match']}"}
    doc = _classification_document("E2", labels, scored["score"].to_numpy(), threshold, meta, inputs, cfg,
                                   extra_counts=extra)
    doc["params"]["iou_positive"] = iou_positive
    return doc


def _match_all(det: "pd.DataFrame", gt: "pd.DataFrame", frames: range, iou: float) -> dict[str, Any]:
    """Match every frame of the frameset; pool TP flags, scores and per-GT found/missed flags."""
    import numpy as np

    flags, scores, gt_found, gt_heights = [], [], [], []
    det_by_frame = dict(tuple(det.groupby("frame")))
    gt_by_frame = dict(tuple(gt.groupby("frame")))
    for frame in frames:
        d = det_by_frame.get(frame)
        g = gt_by_frame.get(frame)
        if d is None and g is None:
            continue
        d_boxes = d[_BOX].to_numpy() if d is not None else np.zeros((0, 4))
        d_scores = d["score"].to_numpy() if d is not None else np.zeros(0)
        g_boxes = g[_BOX].to_numpy() if g is not None else np.zeros((0, 4))
        r = M.match_frame(g_boxes, d_boxes, d_scores, iou)
        flags.append(r["tp_flags"]); scores.append(d_scores)
        gt_found.append(r["gt_matched"]); gt_heights.append(g_boxes[:, 3])
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
    flags, scores = cat(flags, bool), cat(scores, float)
    found, heights = cat(gt_found, bool), cat(gt_heights, float)
    tp = int(flags.sum())
    return {"tp": tp, "fp": int(len(flags) - tp), "fn": int(len(found) - found.sum()), "flags": flags,
            "scores": scores, "gt_found": found, "gt_heights": heights}


def _height_bin_labels(bins: list[float]) -> list[str]:
    labels = [f"{bins[i]:g}-{bins[i + 1]:g}" for i in range(len(bins) - 1)]
    labels[-1] = f"{bins[-2]:g}+"
    return labels


def evaluate_detections(detections_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any],
                        ground_truth: str | None = None) -> dict[str, Any]:
    """E3: restrict GT to the metadata frameset; P/R/F1 at the operating threshold for each IoU threshold;
    AP@0.5 and AP@[.5:.95] over all detections; recall by height bin; FP per frame; FPS from metadata."""
    import numpy as np

    ev = _ev(cfg)
    det = contracts.read_table(detections_csv, "detections")
    meta = contracts.read_run_meta(detections_csv)
    contracts.validate_detections(det, meta)
    frames = _frames(meta)
    gt = _gt_in_frames(gt_tracks, frames)
    threshold = float(meta["operating_threshold"])
    above = det.loc[det["score"] >= threshold]  # P/R/F1 at the threshold the method owner chose on validation
    iou_thresholds = [float(t) for t in ev["detection"]["iou_thresholds"]]

    main = _match_all(above, gt, frames, iou_thresholds[0])
    precision, recall, f1 = M.precision_recall_f1(main["tp"], main["fp"], main["fn"])
    breakdown: dict[str, dict] = {}
    for iou in iou_thresholds[1:]:
        r = _match_all(above, gt, frames, iou)
        p, rc, f = M.precision_recall_f1(r["tp"], r["fp"], r["fn"])
        breakdown[f"at_iou_{iou:g}"] = {"precision": _round(p), "recall": _round(rc), "f1": _round(f)}

    # Recall per GT box height: tiny far-away players are the hard case. Bins are (low, high].
    bins = [float(b) for b in ev["detection"]["height_bins_px"]]
    which = np.digitize(main["gt_heights"], bins[1:-1], right=True)
    breakdown["recall_by_height_px"] = {
        label: (_round(main["gt_found"][which == i].mean()) if (which == i).any() else None)
        for i, label in enumerate(_height_bin_labels(bins))}

    # AP uses ALL detections (>= score floor), never only those above the operating threshold.
    start, stop, step = ev["detection"]["ap_iou_range"]
    ap_ious = np.round(np.arange(start, stop + step / 2, step), 2)
    aps, curve = [], {}
    for iou in ap_ious:
        r = _match_all(det, gt, frames, float(iou))
        ap, pr = M.average_precision(r["flags"], r["scores"], len(gt))
        aps.append(ap)
        if np.isclose(iou, 0.5):
            ap50, curve = ap, pr
    ap50_95 = None if any(a is None for a in aps) else float(np.mean(aps))

    values = {"precision": precision, "recall": recall, "f1": f1, "ap50": ap50, "ap50_95": ap50_95,
              "fp_per_frame": main["fp"] / len(frames)}
    if meta.get("frames_per_second") is not None:
        values["fps"] = float(meta["frames_per_second"])
    counts = {"frames": len(frames), "gt": int(len(gt)), "pred": int(len(above)),
              "tp": main["tp"], "fp": main["fp"], "fn": main["fn"]}
    params = {"iou_threshold": iou_thresholds[0], "score_threshold": threshold, "iou_thresholds": iou_thresholds,
              "score_floor": meta["score_floor"], "matching": ev["detection"]["matching"],
              "ap_interpolation": ev["detection"]["ap_interpolation"],
              "pitch_filter": "off" if "pitch_filter=off" in meta.get("notes", "") else "as produced"}
    inputs = {"prediction": _relative(detections_csv), "ground_truth": ground_truth or f"soccertrack:{meta['match']}"}
    return _document("E3", meta["method"], meta["split"], meta["match"], meta["frameset"], params, counts, values,
                     inputs, cfg, breakdown=breakdown, curves={"pr": curve} if curve["recall"] else None)


def export_trackeval(gt_tracks: "pd.DataFrame", tracks_csv: Path, run_id: str) -> Path:
    """Write the TrackEval workspace (Contracts §9): 1-based frames, GT conf/class/vis = 1, seqinfo.ini.

    Target output for the fixtures: tests/fixtures/trackeval_example/.
    """
    raise NotImplementedError("TODO(Member 3, M3)")


def evaluate_tracking(tracks_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any]) -> dict[str, Any]:
    """E4: export, run TrackEval (HOTA, CLEAR, Identity) with SKIP_SPLIT_FOL=True, return Metrics JSON content."""
    raise NotImplementedError("TODO(Member 3, M3)")


def pool_metrics(per_match: list[dict[str, Any]]) -> dict[str, Any]:
    """Pooled result (match = "pooled"): recompute from summed counts where possible, never average ratios."""
    raise NotImplementedError("TODO(Member 3, M4)")


def write_metrics(metrics: dict[str, Any], out_dir: Path | None = None) -> Path:
    """Validate with contracts.validate_metrics and write outputs/metrics/<run_id>.json."""
    import json

    contracts.validate_metrics(metrics)
    out_dir = Path(out_dir) if out_dir is not None else REPO_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{metrics['run_id']}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path
