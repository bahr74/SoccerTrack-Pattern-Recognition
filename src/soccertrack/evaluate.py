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


def validate_protocol(meta: dict, cfg: dict, kind: str, gt_tracks=None) -> None:
    """Validate producer declarations, not the presence of prediction rows in empty frames.

    Complete clip GT (or an explicit source_num_frames verified by the runner) is
    required to check open-ended framesets. Metadata cannot prove that inference
    actually occurred: producers must retain frame/decode provenance as well.
    """
    from soccertrack import split
    import math
    contracts.validate_run_meta(meta)
    if meta['kind'] != kind:
        raise contracts.ContractError(f'Expected {kind} metadata')
    ev = _ev(cfg)
    from soccertrack import config
    try:
        selected_hash = config.config_sha256()
    except NotImplementedError:
        selected_hash = None  # dependency remains explicit in provenance; no substitute hash
    if ev['detection']['matching'] != 'greedy_by_score' or ev['detection']['ap_interpolation'] != 'all_point':
        raise contracts.ContractError('Unsupported detection matching/AP protocol')
    if ev['detection']['iou_thresholds'] != [.5,.3] or ev['detection']['ap_iou_range'] != [.5,.95,.05]:
        raise contracts.ContractError('Detection IoU thresholds must match the approved protocol')
    fixture = (meta.get('frameset') == 'fixture3' and str(meta.get('match', '')).startswith('999')) or (kind=='patch_scores' and meta.get('device')=='synthetic')
    if not fixture:
        if 'project' not in cfg:
            raise contracts.ContractError('Project split configuration is required')
        if kind == 'patch_scores':
            if meta['match'] is not None or meta['frames'] is not None or meta['frameset'] is not None:
                raise contracts.ContractError('Aggregate patch metadata must have null match/frames/frameset')
        else:
            try:
                actual_split = split.split_of(meta['match'], cfg)
            except ValueError as exc:
                raise contracts.ContractError(str(exc)) from exc
            if actual_split != meta['split']:
                raise PermissionError('Match/split mismatch')
    if not fixture and selected_hash is not None and meta['config_sha256'] != selected_hash:
        raise contracts.ContractError('Source config hash differs from selected shared configuration')
    threshold = meta.get('operating_threshold')
    if threshold is None:
        raise contracts.ContractError('An explicit reporting threshold is required')
    expected = cfg.get('yolo', {}).get('operating_threshold') if meta['method'].startswith('yolo') else cfg.get('scratch', {}).get('detector', {}).get('operating_threshold')
    if meta['method'] in ('gt_oracle','gt_noisy'): expected=None
    if not fixture and kind != 'patch_scores' and meta['method'] not in ('gt_oracle','gt_noisy') and expected is None:
        raise contracts.ContractError('Publish the selected reporting threshold before official evaluation')
    if expected is not None and not math.isclose(threshold, expected, abs_tol=1e-12):
        raise contracts.ContractError('Reporting threshold differs from selected configuration')
    if kind == 'patch_scores':
        return
    name = meta['frameset']
    if name not in ev['framesets']:
        raise contracts.ContractError(f'Unknown frameset {name!r}; debug runs need a distinct configured frameset')
    spec = ev['framesets'][name]
    actual = meta['frames']
    count = meta.get('source_num_frames')
    if count is None and gt_tracks is not None and len(gt_tracks):
        count = int(gt_tracks.frame.max()) + 1
    if count is None and spec['stop'] is None:
        raise contracts.ContractError('Full source frame count is required to certify coverage')
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count <= 0):
        raise contracts.ContractError('Invalid source_num_frames')
    if count is not None and gt_tracks is not None and len(gt_tracks) and count < int(gt_tracks.frame.max())+1:
        raise contracts.ContractError('Declared source frame count truncates the supplied full GT')
    stop = count if spec['stop'] is None else min(spec['stop'], count) if count is not None else spec['stop']
    if actual != dict(start=spec['start'], stop=stop, stride=spec['stride']):
        raise contracts.ContractError('Frameset range/stride/coverage does not match the approved protocol')
    if kind == 'tracks' and actual['stride'] != 1:
        raise contracts.ContractError('Tracking requires stride 1, never sparse eval10')
    if kind == 'detections' and (meta['score_floor'] is None or not math.isclose(meta['score_floor'], ev['score_floor'], abs_tol=1e-12)):
        raise contracts.ContractError('Score floor differs from approved protocol')
    if meta.get('processed_frame_count', len(_frames(meta))) != len(_frames(meta)):
        raise contracts.ContractError('Incomplete processed frame coverage')
    if 'processed_frame_ids' in meta and meta['processed_frame_ids'] != list(_frames(meta)):
        raise contracts.ContractError('Incomplete processed frame IDs')


def _protocol_params(meta, cfg):
    """Lineage retained in every metric document, including runtime measurement scope."""
    return {'source_config_sha256': meta['config_sha256'], 'frames': meta['frames'],
            'model_sha256':meta.get('model_sha256',meta.get('weights_sha256')),
            'tracker_settings':meta.get('tracker_settings'),
            'score_floor': meta['score_floor'], 'score_threshold': meta['operating_threshold'],
            'runtime_scope': meta.get('runtime_scope', 'unspecified; see source notes'),
            'source_notes': meta['notes'],
            'result_scope': 'synthetic' if meta.get('frameset') == 'fixture3' or str(meta.get('match', '')).startswith('999') else 'protocol_validated',
            'metric_definition': 'classification_average_precision' if meta['kind'] in ('patch_scores', 'proposals') else 'detection_AP50_AP50:95' if meta['kind'] == 'detections' else 'TrackEval'}


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
    provenance = _provenance(cfg)
    if provenance['config_sha256'] == '0'*64 and params.get('result_scope') != 'synthetic':
        params = dict(params, result_scope='integration_pending')
    doc = {
        "schema_version": contracts.SCHEMA_VERSION,
        "run_id": f"{eval_id}__{method}__{split}__{match}__{frameset}",
        "eval_id": eval_id, "task": contracts.EVAL_TASKS[eval_id], "method": method, "split": split,
        "match": match, "frameset": frameset, "params": params, "counts": counts,
        "metrics": {k: (_round(v) if isinstance(v, float) else v) for k, v in values.items()},
        "inputs": inputs, "provenance": provenance,
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
                     _protocol_params(meta, cfg), counts, result["metrics"], inputs, cfg,
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
    if not np.isfinite(drop_rate) or not 0 <= drop_rate <= 1 or not np.isfinite(jitter_px) or jitter_px < 0:
        raise ValueError('drop_rate must be in [0,1] and jitter_px finite/nonnegative')
    if image_size is not None and (len(image_size) != 2 or any(not np.isfinite(v) or v < 2 for v in image_size)):
        raise ValueError('Image must accommodate boxes at least 2 pixels wide/high')
    keep = rng.random(len(gt_tracks)) >= drop_rate
    det = gt_tracks.loc[keep, list(contracts.DETECTION_COLUMNS)].copy().reset_index(drop=True)
    boxes = det[_BOX].to_numpy(dtype=float) + rng.normal(0.0, jitter_px, (len(det), 4))
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    x1, y1 = np.maximum(x1, 0.0), np.maximum(y1, 0.0)
    if image_size is not None:
        x1, y1 = np.minimum(x1, image_size[0] - 2.0), np.minimum(y1, image_size[1] - 2.0)
    x2, y2 = np.maximum(x2, x1 + 2.0), np.maximum(y2, y1 + 2.0)
    if image_size is not None:
        x2, y2 = np.minimum(x2, image_size[0]), np.minimum(y2, image_size[1])
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
    validate_protocol(meta, cfg, 'patch_scores')
    from soccertrack import split
    synthetic = meta.get('device') == 'synthetic' and table['match'].str.startswith('999').all()
    if not synthetic and not set(table['match']).issubset(split.matches_for(meta['split'], cfg)):
        raise PermissionError('Patch score rows contain a match outside the declared split')
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
    validate_protocol(meta, cfg, 'proposals', gt_tracks)
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
    # Coverage is many-to-one (proposal upper bound), separate from classification
    # labels and confidence-greedy one-to-one E3 matching.
    recalls = {}
    for t in _ev(cfg)['detection']['iou_thresholds']:
        found = 0
        for f, truth in gt.groupby('frame'):
            boxes = table.loc[table.frame == f, _BOX].to_numpy()
            if len(boxes):
                found += int((M.iou_matrix(truth[_BOX].to_numpy(),boxes).max(axis=1) >= t).sum())
        recalls[str(t)] = _round(found/len(gt)) if len(gt) else None
    doc['breakdown'] = {'proposal_recall_at_iou': recalls}
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
    validate_protocol(meta, cfg, 'detections', gt_tracks)
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
    params.update(_protocol_params(meta, cfg))
    params['ap_iou_range'] = list(ev['detection']['ap_iou_range'])
    inputs = {"prediction": _relative(detections_csv), "ground_truth": ground_truth or f"soccertrack:{meta['match']}"}
    return _document("E3", meta["method"], meta["split"], meta["match"], meta["frameset"], params, counts, values,
                     inputs, cfg, breakdown=breakdown, curves={"pr": curve} if curve["recall"] else None)


def _sequence_name(meta: dict[str, Any]) -> str:
    return f"{meta['match']}__{meta['frameset']}"


def export_trackeval(gt_tracks: "pd.DataFrame", tracks_csv: Path, run_id: str,
                     workspace_root: Path | None = None) -> Path:
    """Write the TrackEval workspace (Contracts §9): 1-based frames, GT conf/class/vis = 1, seqinfo.ini.

    Target output for the fixtures: tests/fixtures/trackeval_example/.
    TrackEval expects the MOTChallenge text format, so this is the ONLY place where frames become 1-based:
      frame_te = frame - frames.start + 1
    Layout: <ws>/gt/<seq>/gt/gt.txt, <ws>/gt/<seq>/seqinfo.ini, <ws>/trackers/<pipeline>/data/<seq>.txt
    """
    tracks = contracts.read_table(tracks_csv, "tracks")
    meta = contracts.read_run_meta(tracks_csv)
    if meta["kind"] != "tracks":
        raise contracts.ContractError(f"{tracks_csv} is a '{meta['kind']}' artefact, not 'tracks'")
    frames = _frames(meta)
    if frames.step != 1:
        raise contracts.ContractError("tracking framesets must have stride 1 (Contracts §9)")
    contracts.validate_tracks(tracks, image_size=(meta["image_width"], meta["image_height"]))
    _check_frames(tracks, frames, "tracks")
    gt = _gt_in_frames(gt_tracks, frames)
    pipeline = contracts.pipeline_id(meta["method"], meta["tracker"])
    seq = _sequence_name(meta)
    ws = (Path(workspace_root) if workspace_root is not None else REPO_ROOT / "outputs" / "trackeval") / run_id

    def lines(table, tail):
        out = []
        for row in table.itertuples(index=False):
            frame_te = int(row.frame) - frames.start + 1
            out.append(f"{frame_te},{int(row.track_id)},{row.x:.2f},{row.y:.2f},{row.w:.2f},{row.h:.2f},{tail(row)}")
        return "".join(line + "\n" for line in out)

    gt_dir = ws / "gt" / seq
    (gt_dir / "gt").mkdir(parents=True, exist_ok=True)
    _write_text(gt_dir / "gt" / "gt.txt", lines(gt, lambda r: "1,1,1"))
    fps = meta["fps"]
    _write_text(gt_dir / "seqinfo.ini", "[Sequence]\n" f"name={seq}\n" "imDir=img1\n"
                f"frameRate={int(fps) if float(fps).is_integer() else fps}\n" f"seqLength={len(frames)}\n"
                f"imWidth={meta['image_width']}\n" f"imHeight={meta['image_height']}\n" "imExt=.jpg\n")
    trk_dir = ws / "trackers" / pipeline / "data"
    trk_dir.mkdir(parents=True, exist_ok=True)
    _write_text(trk_dir / f"{seq}.txt", lines(tracks, lambda r: f"{r.score:.4f},-1,-1,-1"))
    return ws


def _write_text(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def _check_frames(table: "pd.DataFrame", frames: range, kind: str) -> None:
    f = table["frame"]
    if len(f) and ((f < frames.start) | (f >= frames.stop) | ((f - frames.start) % frames.step != 0)).any():
        raise contracts.ContractError(f"{kind}: frames outside the metadata frameset {frames}")


def evaluate_tracking(tracks_csv: Path, gt_tracks: "pd.DataFrame", cfg: dict[str, Any],
                      workspace_root: Path | None = None, ground_truth: str | None = None) -> dict[str, Any]:
    """E4: export, run TrackEval (HOTA, CLEAR, Identity) with SKIP_SPLIT_FOL=True, return Metrics JSON content.

    HOTA = sqrt(DetA x AssA), averaged over IoU thresholds 0.05 ... 0.95 (TrackEval's alpha grid):
      DetA  "are the players detected?"       AssA "do the IDs stay with the same player over time?"
      LocA  "how well do the boxes fit?"
    MOTA = 1 - (FN + FP + IDSW) / number of GT boxes   (can be negative)
    IDF1 = 2 IDTP / (2 IDTP + IDFP + IDFN)            (identity-level F1 over the whole sequence)
    IDSW = number of times a GT player's predicted ID changes.
    """
    import contextlib
    import io

    import numpy as np
    import trackeval

    meta = contracts.read_run_meta(tracks_csv)
    validate_protocol(meta, cfg, 'tracks', gt_tracks)
    pipeline = contracts.pipeline_id(meta["method"], meta["tracker"])
    run_id = f"E4__{pipeline}__{meta['split']}__{meta['match']}__{meta['frameset']}"
    ws = export_trackeval(gt_tracks, tracks_csv, run_id, workspace_root)
    seq = _sequence_name(meta)
    frames = _frames(meta)

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False,
                        "PRINT_CONFIG": False, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True,
                        "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False,
                        "LOG_ON_ERROR": None, "BREAK_ON_ERROR": True})
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({"GT_FOLDER": str(ws / "gt"), "TRACKERS_FOLDER": str(ws / "trackers"),
                           "OUTPUT_FOLDER": str(ws / "results"), "TRACKERS_TO_EVAL": [pipeline],
                           "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
                           "SEQ_INFO": {seq: len(frames)}, "SKIP_SPLIT_FOL": True, "DO_PREPROC": True,
                           "PRINT_CONFIG": False, "TRACKER_SUB_FOLDER": "data"})
    threshold = float(_ev(cfg).get("tracking", {}).get("similarity_threshold", 0.5))
    metric_objects = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR({"THRESHOLD": threshold, "PRINT_CONFIG": False}),
                      trackeval.metrics.Identity({"THRESHOLD": threshold, "PRINT_CONFIG": False})]
    with contextlib.redirect_stdout(io.StringIO()):  # TrackEval prints a lot; keep the console clean
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        results, messages = evaluator.evaluate([dataset], metric_objects)
    res = results["MotChallenge2DBox"][pipeline][seq]["pedestrian"]
    hota, clear, ident = res["HOTA"], res["CLEAR"], res["Identity"]

    tracks = contracts.read_table(tracks_csv, "tracks")
    gt = _gt_in_frames(gt_tracks, frames)
    values = {"hota": float(np.mean(hota["HOTA"])), "deta": float(np.mean(hota["DetA"])),
              "assa": float(np.mean(hota["AssA"])), "loca": float(np.mean(hota["LocA"])),
              "idf1": float(ident["IDF1"]), "mota": float(clear["MOTA"]), "idsw": int(clear["IDSW"])}
    counts = {"frames": len(frames), "gt_ids": int(gt["track_id"].nunique()),
              "pred_ids": int(tracks["track_id"].nunique()), "gt": int(len(gt)), "pred": int(len(tracks)),
              "tp": int(clear["CLR_TP"]), "fn": int(clear["CLR_FN"]), "fp": int(clear["CLR_FP"]),
              "idtp": int(ident["IDTP"]), "idfn": int(ident["IDFN"]), "idfp": int(ident["IDFP"])}
    params = {"similarity_threshold": threshold, "trackeval_metrics": ["HOTA", "CLEAR", "Identity"],
              "hota_alphas": "0.05:0.05:0.95 (mean)", "sequence": seq}
    params.update(_protocol_params(meta, cfg))
    inputs = {"prediction": _relative(tracks_csv), "ground_truth": ground_truth or f"soccertrack:{meta['match']}",
              "trackeval_workspace": _relative(ws)}
    # Store unrounded sufficient statistics in the existing open inputs object.
    # Scalar averaged HOTA cannot be combined exactly later.
    def native(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
    inputs['trackeval_statistics'] = {name: {key: native(value) for key, value in result.items()}
                                     for name, result in [('HOTA', hota), ('CLEAR', clear), ('Identity', ident)]}
    return _document("E4", pipeline, meta["split"], meta["match"], meta["frameset"], params, counts, values,
                     inputs, cfg)


def pool_metrics(per_match: list[dict[str, Any]], ap: dict[str, float | None] | None = None) -> dict[str, Any]:
    """Pooled result (match = "pooled"): recompute from summed counts where possible, never average ratios.

    Why: averaging ratios gives a small match the same weight as a big one. Summing TP/FP/FN first and then
    dividing weights every box equally.
      E1/E2  sum TP/FP/TN/FN -> sensitivity, specificity, precision, F1, accuracy. ROC-AUC/PR-AUC need the raw
             scores, so they are null here (E1 is already computed on the whole split).
      E3     sum frames/GT/pred/TP/FP/FN -> P, R, F1, FP per frame. AP needs every detection pooled, so it is
             passed in through `ap` (see pooled_detection_ap); otherwise null.
      E4     TrackEval combine_sequences from unrounded per-sequence HOTA/CLEAR/Identity statistics.
             Legacy scalar-only inputs are rejected rather than approximated.
    """
    if not per_match:
        raise ValueError("pool_metrics needs at least one per-match Metrics JSON")
    first = per_match[0]
    for doc in per_match:
        contracts.validate_metrics(doc)
        if doc['match'] == 'pooled':
            raise ValueError('Pool original per-match runs only, never nested pooled summaries')
        for key in ("eval_id", "method", "split", "frameset"):
            if doc[key] != first[key]:
                raise ValueError(f"cannot pool runs with different {key}: {first[key]} vs {doc[key]}")
        ignored = {'sequence', 'frames', 'source_notes', 'source_num_frames', 'pooling', 'matches'}
        protocol = lambda d: {k: v for k, v in d['params'].items() if k not in ignored}
        if protocol(doc) != protocol(first):
            raise ValueError('cannot pool incompatible evaluation protocols/configurations')
        ranges = [d['params'].get('frames') for d in (first,doc)]
        signature = lambda r: None if r is None else (r['start'],r['stride'])
        if signature(ranges[0]) != signature(ranges[1]):
            raise ValueError('cannot pool incompatible frame range/stride protocols')
    if len({doc["match"] for doc in per_match}) != len(per_match):
        raise ValueError("each match may appear only once in a pooled result")
    task = first["task"]
    counts: dict[str, int] = {}
    for doc in per_match:
        for key, value in doc["counts"].items():
            counts[key] = counts.get(key, 0) + int(value)
    c = counts
    if task in ("patch_classification", "proposal_classification"):
        precision, sensitivity, f1 = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
        total = c["tp"] + c["fp"] + c["tn"] + c["fn"]
        values = {"sensitivity": sensitivity, "specificity": c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0,
                  "precision": precision, "f1": f1, "accuracy": (c["tp"] + c["tn"]) / total if total else 0.0,
                  "roc_auc": None, "pr_auc": None}
        method = "sum of counts; ROC-AUC and PR-AUC need pooled scores"
    elif task == "detection":
        precision, recall, f1 = M.precision_recall_f1(c["tp"], c["fp"], c["fn"])
        values = {"precision": precision, "recall": recall, "f1": f1,
                  "ap50": (ap or {}).get("ap50"), "ap50_95": (ap or {}).get("ap50_95"),
                  "fp_per_frame": c["fp"] / c["frames"] if c["frames"] else 0.0}
        method = "sum of counts; AP from detections pooled across matches" if ap else "sum of counts; AP not pooled"
    elif task == "tracking":
        import numpy as np
        import trackeval
        combined = {}
        for name, metric in [('HOTA', trackeval.metrics.HOTA()),
                             ('CLEAR', trackeval.metrics.CLEAR({'PRINT_CONFIG': False})),
                             ('Identity', trackeval.metrics.Identity({'PRINT_CONFIG': False}))]:
            sequences = {}
            for doc in per_match:
                stats = doc['inputs'].get('trackeval_statistics', {}).get(name)
                if not isinstance(stats, dict):
                    raise ValueError('Exact tracking pooling requires unrounded TrackEval sufficient statistics; reevaluate legacy inputs')
                sequences[doc['match']] = {k: np.asarray(v) if isinstance(v, list) else v for k, v in stats.items()}
            combined[name] = metric.combine_sequences(sequences)
        values = {key: float(np.mean(combined['HOTA'][field]))
                  for key, field in [('hota', 'HOTA'), ('deta', 'DetA'), ('assa', 'AssA'), ('loca', 'LocA')]}
        values.update(idsw=int(combined['CLEAR']['IDSW']), mota=float(combined['CLEAR']['MOTA']),
                      idf1=float(combined['Identity']['IDF1']))
        method = 'TrackEval combine_sequences using unrounded HOTA/CLEAR/Identity statistics'
    else:
        raise ValueError(f"unknown task {task}")
    params = dict(first["params"], pooling=method, matches=sorted(doc["match"] for doc in per_match))
    params.pop('frames',None)
    inputs = {"per_match_run_ids": [doc["run_id"] for doc in per_match],
              'frames_by_match': {doc['match']:doc['params'].get('frames') for doc in per_match},
              'source_provenance': {doc['match']:doc['provenance'] for doc in per_match}}
    return _document(first["eval_id"], first["method"], first["split"], "pooled", first["frameset"], params,
                     counts, values, inputs, {})


def pooled_detection_ap(runs: list[tuple[Path, "pd.DataFrame"]], cfg: dict[str, Any]) -> dict[str, float | None]:
    """AP@0.5 and AP@[.5:.95] with the detections of several matches pooled into ONE ranked list.

    runs = [(detections_csv, gt_tracks_of_that_match), ...]. AP is not a sum of counts, so it cannot be pooled
    from the per-match JSON files; it is recomputed here from the raw tables.
    """
    import numpy as np

    ev = _ev(cfg)
    start, stop, step = ev["detection"]["ap_iou_range"]
    loaded = []
    first_protocol = None
    for csv_path, gt_tracks in runs:
        det = contracts.read_table(csv_path, "detections")
        meta = contracts.read_run_meta(csv_path)
        validate_protocol(meta, cfg, 'detections', gt_tracks)
        contracts.validate_detections(det, meta)
        protocol = {k: meta[k] for k in ('method','split','frameset','score_floor','operating_threshold','config_sha256')}
        if first_protocol is not None and protocol != first_protocol:
            raise ValueError('Cannot pool AP from incompatible producer protocols')
        first_protocol = protocol
        frames = _frames(meta)
        loaded.append((det, _gt_in_frames(gt_tracks, frames), frames))
    aps = {}
    for iou in np.round(np.arange(start, stop + step / 2, step), 2):
        flags, scores, num_gt = [], [], 0
        for det, gt, frames in loaded:
            r = _match_all(det, gt, frames, float(iou))
            flags.append(r["flags"]); scores.append(r["scores"]); num_gt += len(gt)
        aps[float(iou)] = M.average_precision(np.concatenate(flags), np.concatenate(scores), num_gt)[0]
    values = list(aps.values())
    return {"ap50": aps.get(0.5), "ap50_95": None if any(v is None for v in values) else float(np.mean(values))}


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
