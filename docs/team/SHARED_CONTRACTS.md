# Shared Integration Contracts (v1.0)

**Status:** frozen for Phase 2 and Phase 3. Every module communicates **only** through the artefacts defined here.
**Owner of this document and of `src/soccertrack/contracts.py`:** Member 3. Changing a contract needs a pull request approved by **all four members** and a version bump (`SCHEMA_VERSION`).
**Technical baseline:** [`docs/planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1).

Every algorithm reads and writes the formats below. Nothing else (hidden column orders, implicit units, private file names) is allowed between members.

---

## 1. Global conventions

| Topic | Rule |
|---|---|
| **Match ID** | String of digits, e.g. `"117093"`. Always quoted in YAML and JSON so it is never parsed as an integer |
| **Frame index** | **0-based** integer. Frame `k` = the k-th frame obtained by **sequential decoding** of the clip, starting at 0. Never seek by timestamp (clip 117093 has a 0.979 s container start offset). The raw SoccerTrack MOT files are already 0-based. All internal tables are 0-based; only the TrackEval exporter (Section 9) converts to 1-based |
| **Image** | Frames are `numpy.ndarray`, `uint8`, shape `(H, W, 3)`, **BGR** (OpenCV order), at the **original clip resolution** (e.g. 4096 × 1080). Patches are **RGB** (Section 6). Conversion happens once, in `patches.crop_patch` |
| **Box coordinates** | Pixels in the **original full-resolution frame**. `x`, `y` = top-left corner; `w`, `h` = width and height. x grows to the right, y grows downwards, origin at the top-left corner of the image. Floats; written with 2 decimals |
| **Box validity** | `w > 1`, `h > 1`, `0 ≤ x`, `0 ≤ y`, `x + w ≤ W`, `y + h ≤ H` (W, H from the run metadata). Producers clip boxes to the image before writing. Boxes produced at a reduced scale (e.g. 0.5 for background subtraction, or tile coordinates) are converted to full-frame coordinates **before** writing |
| **Score** | Float in [0, 1], 4 decimals. Higher = more likely a player. GT rows use `1.0` |
| **Track ID** | Positive integer, unique within one (pipeline, match, frameset). GT uses the original SoccerTrack IDs (1–22) |
| **Units** | Pixels for image space; metres, seconds, m/s and m/s² for pitch space (Phase 3) |
| **Metric values** | Bounded ratios (precision, recall, F1, AP, sensitivity, specificity, accuracy, ROC-AUC, PR-AUC, HOTA, DetA, AssA, LocA, IDF1) are fractions in [0, 1] (e.g. HOTA 0.42, never 42). **MOTA** is a fraction ≤ 1 that **can be negative**. `fp_per_frame` and `fps` are non-negative reals that can exceed 1. `idsw` and all `counts` are non-negative integers. Figures may display percentages |
| **Tables** | CSV, UTF-8, comma separator, header row, no index column, `\n` line endings. Read and written only through `contracts.read_table` / `contracts.write_table` |
| **JSON** | UTF-8, 2-space indent, keys in snake_case, timestamps in ISO-8601 UTC (`2026-11-21T14:03:00Z`) |
| **Paths in files** | Repository-relative POSIX paths (`outputs/detections/...`). Never absolute paths, except in the git-ignored `configs/data_sources.yaml` |

### 1.1 Method, tracker and pipeline identifiers (closed vocabulary)

Defined as constants in `contracts.py`. New IDs require a contract change.

| Kind | ID | Owner | Phase |
|---|---|---|---|
| Detector | `gt_oracle` (GT boxes converted to detections, score 1.0) | Member 3 | 2 |
| Detector | `gt_noisy` (GT with simulated drops and jitter) | Member 3 | 2 |
| Detector | `scratch_mwb0` (rule-based verifier) | Member 1 | 2 |
| Detector | `scratch_hog_svm` | Member 1 | 2 |
| Detector | `scratch_cnn` | Member 1 | 2 |
| Detector | `yolov8n_coco` (zero-shot) | Member 2 | 2 |
| Detector | `yolov8n_ft`, `yolov8s_ft` (fine-tuned) | Member 2 | 3 |
| Detector | `scratch_hog_svm_v2`, `scratch_cnn_v2` (improved) | Member 1 | 3 |
| Tracker | `iou_hungarian` (from scratch) | Member 1 | 2 |
| Tracker | `bytetrack` (Ultralytics BYTETracker adapter) | Member 2 | 2 |
| Tracker | `iou_hungarian_v2` (Kalman / two-stage), `bytetrack_tuned` | Member 1 / Member 2 | 3 |

**Pipeline ID** = `<detector>+<tracker>`, e.g. `scratch_hog_svm+iou_hungarian`, `yolov8n_coco+bytetrack`, `gt_oracle+iou_hungarian`, `gt_oracle+bytetrack`.

The two official end-to-end pipelines are **A: `scratch_*+iou_hungarian`** and **B: `yolov8n_*+bytetrack`**. Both trackers are also run on `gt_oracle` and `gt_noisy` for the oracle evaluation. They remain two separate implementations.

### 1.2 Framesets (which frames a run covers)

Defined in `configs/evaluation.yaml` (Member 3). Producers read them; they never invent their own frame selections for evaluated runs.

| Frameset | Definition | Used for |
|---|---|---|
| `eval10` | `range(0, num_frames, 10)` | E2, E3 detection evaluation (Phase 2 and 3) |
| `win60s` | `range(0, 1500, 1)` (first 60 s at 25 fps) | E4 tracking in Phase 2 |
| `full` | `range(0, num_frames, 1)` | E4 in Phase 3; workload |
| `smoke` | `range(0, 100, 1)` | Integration smoke tests only; never reported |
| `fixture3` | `range(0, 3, 1)` | Synthetic test fixtures only |

`num_frames` = decoded frame count of the clip (equal to the number of annotated frames; verified by `scripts/check_data.py`).

---

## 2. Output locations and file naming

All under `outputs/` (git-ignored). `<split>` ∈ {`train`, `val`, `test`}.

| Artefact | Path | Producer |
|---|---|---|
| Detection table + metadata | `outputs/detections/<detector>/<split>/<match>__<frameset>.csv` + `.meta.json` | Member 1, Member 2, Member 3 (oracle) |
| Track table + metadata | `outputs/tracks/<pipeline>/<split>/<match>__<frameset>.csv` + `.meta.json` | Member 1, Member 2 |
| Proposal table (scored) + metadata | `outputs/proposals/<detector>/<split>/<match>__<frameset>.csv` + `.meta.json` | Member 1 |
| Patch set | `data/processed/patches/patches_<split>.npz` | Member 1 |
| Patch score table + metadata | `outputs/scores/<detector>/<split>/patches.csv` + `.meta.json` | Member 1 |
| Pitch mask | `outputs/masks/<match>_pitch.png` | Member 1 |
| Background image | `outputs/backgrounds/<match>.png` | Member 1 |
| Segmentation stage images | `outputs/stages/<match>/<frame>_{frame,foreground,morph,proposals}.png` | Member 1 |
| EDA statistics | `outputs/eda/stats.json`, `outputs/eda/box_sample.csv` | Member 1 |
| Trained models | `outputs/models/<detector>/` | Member 1, Member 2 |
| Metrics JSON | `outputs/metrics/<run_id>.json` | Member 3 |
| Comparison tables | `outputs/tables/<name>.csv` | Member 3 |
| Figures | `outputs/figures/<owner_area>/<name>.png` (`eda`, `eval`, `overlays`, `workload`) | Member 3 (`eval`), Member 4 (others) |
| TrackEval workspace | `outputs/trackeval/<run_id>/` | Member 3 |
| Test-access log | `outputs/test_log.md` | appended by runners via `split.log_test_access` |
| Pitch tracks (Phase 3) | `outputs/pitch/<pipeline>/<split>/<match>__<frameset>.csv` | Member 4 |
| Workload summary (Phase 3) | `outputs/workload/<pipeline>/<split>/<match>__<frameset>.csv` | Member 4 |

Curated figures for the report are copied by Member 4 into `docs/phase2/figures/` or `docs/phase3/figures/`. These are small PNGs and are committed.

---

## 3. Track table

Ground truth and every tracker output use this one schema.

| Column | Type | Rule |
|---|---|---|
| `frame` | int | 0-based, inside the run's frameset |
| `track_id` | int | > 0 |
| `x`, `y`, `w`, `h` | float | Box convention of Section 1 |
| `score` | float | [0, 1]; GT = 1.0; trackers write the score of the associated detection (ByteTrack: its track score) |

**Validation rules (`contracts.validate_tracks`):**
- exact column names and order;
- no NaN;
- no duplicate (`frame`, `track_id`) pairs;
- box validity against the image size, if given;
- rows sorted by `frame`, then `track_id`.

**Example** (`tests/fixtures/tracks_gt.csv`):
```
frame,track_id,x,y,w,h,score
0,1,100.00,150.00,20.00,45.00,1.0000
0,2,300.00,160.00,20.00,45.00,1.0000
0,3,500.00,140.00,18.00,40.00,1.0000
```

### 3.1 Raw SoccerTrack MOT input (read only by `data_io`)
- 10 columns, no header: `frame,id,x,y,w,h,c7,c8,c9,c10`.
- Floats with 2 decimals.
- All four trailing columns are `-1` in every row of the 6 locally checked train/val files (verified 2026-11-19).
- `data_io.read_mot_raw` asserts the column count; `data_io.mot_to_tracks` converts to the Track table: `track_id = id`, `score = 1.0`. It also reports (and does not silently fix):
  - degenerate boxes;
  - boxes outside the image (clipped);
  - duplicate rows;
  - frames whose box count differs from `expected_boxes_per_frame` (22 for SoccerTrack; 3 in the fixtures).

---

## 4. Detection table

| Column | Type | Rule |
|---|---|---|
| `frame` | int | 0-based, inside the run's frameset |
| `x`, `y`, `w`, `h` | float | Box convention of Section 1, full-frame coordinates |
| `score` | float | [0, 1], ≥ `score_floor` from the metadata |

**Producer obligations:**
- Final NMS is already applied (use `torchvision.ops.nms`).
- Write **every** detection with `score ≥ score_floor`. The default floor is 0.05, set in `configs/evaluation.yaml`. Do **not** pre-filter at the operating threshold, because AP, PR curves and ByteTrack's low-score association need the low-score detections.
- Frames with zero detections simply have no rows. The metadata frameset tells the evaluator which frames were processed.
- Sort rows by `frame`, then `score` descending.

**Validation (`contracts.validate_detections`):** column names and order, no NaN, score range, box validity, and every frame inside the frameset declared in the metadata.

**Example** (`tests/fixtures/detections_sample.csv`):
```
frame,x,y,w,h,score
0,101.00,150.00,20.00,45.00,0.9200
0,299.00,161.00,20.00,45.00,0.8800
0,50.00,50.00,20.00,45.00,0.4000
```

---

## 5. Proposal table (Approach A only)

| Column | Type | Rule |
|---|---|---|
| `frame` | int | 0-based |
| `x`, `y`, `w`, `h` | float | Full-frame coordinates |
| `area` | int | Foreground pixel count of the connected component (full-frame scale) |
| `fill_ratio` | float | `area / (w * h)`, in (0, 1] |
| `score` | float | Classifier score in [0, 1] for **scored** proposals; `-1.0` for **unscored** proposals |

Proposals are stored **before NMS** so that E2 sees the true candidate distribution.

**Example** (`tests/fixtures/proposals_sample.csv`): see the fixture file.

---

## 6. Patch set (Approach A classifiers)

A NumPy `.npz` archive written with `numpy.savez_compressed`.

| Key | dtype / shape | Meaning |
|---|---|---|
| `X` | `uint8 [N, 64, 32, 3]` | **RGB** patch, 64 rows (height) × 32 columns (width) |
| `y` | `int8 [N]` | 1 = player, 0 = background |
| `match` | `<U6 [N]` | Match ID |
| `frame` | `int32 [N]` | Source frame |
| `box` | `float32 [N, 4]` | Source box `x, y, w, h` (before the 10 % context margin) |
| `kind` | `<U8 [N]` | `gt`, `random`, `partial`, `proposal` |
| `meta_json` | `<U [1]` | JSON string: split, ratio, seed, sampling stride, IoU bounds, `schema_version` |

**Rules:**
- Positives = GT boxes; negatives = `random` (IoU < 0.1 with every GT box, inside the pitch mask), `partial` (0.1 ≤ IoU < 0.4), `proposal` (unmatched proposals).
- Negative:positive ratio = 3:1 (documented in `meta_json`).
- One file per split.
- Test patches are created only during the frozen test run.

**Model input/output expectations (all patch classifiers):**
- Input: `X` as above (`uint8`, RGB, 64×32).
- Each classifier performs its own normalisation internally (CNN mean/std computed on the training split only; HOG on grayscale).
- Output: `predict_proba(X) -> float32 [N]` in [0, 1].

### 6.1 Patch score table (input to E1)

| Column | Type |
|---|---|
| `sample_id` | int (row index in the patch set) |
| `match` | str |
| `frame` | int |
| `kind` | str |
| `y_true` | int {0, 1} |
| `score` | float [0, 1] |

The metadata field `operating_threshold` gives the threshold chosen on validation.

---

## 7. Run metadata sidecar (`<same name>.meta.json`)

Required next to every Detection, Track, Proposal and Patch score table.

```json
{
  "schema_version": "1.0",
  "kind": "detections",
  "method": "yolov8n_coco",
  "tracker": null,
  "split": "val",
  "match": "117093",
  "frameset": "eval10",
  "frames": {"start": 0, "stop": 5976, "stride": 10},
  "image_width": 4096,
  "image_height": 1080,
  "fps": 25.0,
  "score_floor": 0.05,
  "operating_threshold": 0.40,
  "frozen_test": false,
  "config_sha256": "<64 hex characters>",
  "git_commit": null,
  "created_utc": "2026-11-21T14:03:00Z",
  "runtime_seconds": 151.2,
  "frames_per_second": 3.95,
  "device": "cuda:0 (GTX 1650 Ti)",
  "notes": ""
}
```

**Field rules:**
- `kind` ∈ {`detections`, `tracks`, `proposals`, `patch_scores`}.
- `frames.stop` is exclusive; the processed frames are exactly `range(start, stop, stride)`.
- `tracker` is a tracker ID for `tracks`, otherwise `null`.
- `operating_threshold` is chosen **on validation** by the method owner and copied unchanged into test runs.
- `frozen_test` must be `true` for every test-split artefact.
- `config_sha256` is computed by `config.config_sha256()`.

---

## 8. Metrics JSON (`outputs/metrics/<run_id>.json`)

`run_id = <eval_id>__<method or pipeline>__<split>__<match or "pooled">__<frameset>`, e.g. `E3__yolov8n_coco__val__117093__eval10`.

```json
{
  "schema_version": "1.0",
  "run_id": "E3__yolov8n_coco__val__117093__eval10",
  "eval_id": "E3",
  "task": "detection",
  "method": "yolov8n_coco",
  "split": "val",
  "match": "117093",
  "frameset": "eval10",
  "params": {"iou_threshold": 0.5, "score_threshold": 0.40},
  "counts": {"frames": 598, "gt": 13156, "pred": 12400, "tp": 10200, "fp": 2200, "fn": 2956},
  "metrics": {"precision": 0.8226, "recall": 0.7753, "f1": 0.7983, "ap50": 0.8010, "ap50_95": 0.3550, "fp_per_frame": 3.68},
  "breakdown": {"recall_by_height_px": {"0-30": 0.41, "30-45": 0.72, "45-60": 0.88, "60+": 0.93}},
  "curves": {"pr": {"recall": [0.0, 0.5, 0.77], "precision": [1.0, 0.95, 0.82]}},
  "inputs": {"prediction": "outputs/detections/yolov8n_coco/val/117093__eval10.csv",
             "ground_truth": "soccertrack:117093"},
  "provenance": {"config_sha256": "<hex>", "git_commit": null, "created_utc": "2026-11-21T15:00:00Z",
                 "package_versions": {"numpy": "2.5.4", "trackeval": "1.3.0"}, "seed": 42}
}
```

(The numbers above are illustrative only.)

**Required keys:** `schema_version`, `run_id`, `eval_id`, `task`, `method`, `split`, `match`, `frameset`, `params`, `counts`, `metrics`, `inputs`, `provenance`. `breakdown` and `curves` are optional.

**Required metric keys by task:**

| `eval_id` / `task` | `metrics` keys | `counts` keys |
|---|---|---|
| E1 `patch_classification` | `sensitivity`, `specificity`, `precision`, `f1`, `accuracy`, `roc_auc`, `pr_auc` | `tp`, `fp`, `tn`, `fn` |
| E2 `proposal_classification` | same as E1 | same as E1 + `proposals`, `gt` |
| E3 `detection` | `precision`, `recall`, `f1`, `ap50`, `ap50_95`, `fp_per_frame` (+ `fps` if known) | `frames`, `gt`, `pred`, `tp`, `fp`, `fn` |
| E4 `tracking` | `hota`, `deta`, `assa`, `loca`, `idf1`, `mota`, `idsw` | `frames`, `gt_ids`, `pred_ids`, `gt`, `pred` |

**Value domains (checked by `validate_metrics`):** see Section 1; the per-key domains are the constants `BOUNDED_METRIC_KEYS`, `MOTA_KEYS`, `NONNEGATIVE_REAL_KEYS` and `INTEGER_METRIC_KEYS` in `contracts.py`. A metric that is undefined (e.g. a height bin without GT) is `null`, never 0.

`curves` (optional) may contain `pr` and `roc` as parallel arrays (≤ 200 points). The confusion matrix is always `counts.tp/fp/tn/fn` for E1 and E2.

---

## 9. TrackEval adapter (export) requirements

Implemented only in `evaluate.export_trackeval` (Member 3). No other module writes TrackEval files.

| Item | Rule |
|---|---|
| Layout | `outputs/trackeval/<run_id>/gt/<seq>/gt/gt.txt`, `outputs/trackeval/<run_id>/gt/<seq>/seqinfo.ini`, `outputs/trackeval/<run_id>/trackers/<pipeline>/data/<seq>.txt`; evaluator option `SKIP_SPLIT_FOL=True` |
| Sequence name | `<match>__<frameset>` |
| Frame conversion | `frame_te = frame - frames.start + 1` (1-based, contiguous). Tracking framesets must have stride 1 |
| GT line (9 columns) | `frame_te,track_id,x,y,w,h,1,1,1` (conf flag 1 = consider, class 1 = pedestrian, visibility 1) — replaces the raw `-1` values |
| Tracker line (10 columns) | `frame_te,track_id,x,y,w,h,score,-1,-1,-1` |
| `seqinfo.ini` | `[Sequence]` with `name`, `imDir=img1`, `frameRate`, `seqLength` (= number of frames in the frameset), `imWidth`, `imHeight`, `imExt=.jpg` |
| Sanity check | GT exported as the tracker gives HOTA = IDF1 = MOTA = 1.0 (unit test) |

The examples are in `tests/fixtures/trackeval_example/` (computed from `tracks_gt.csv`).

---

## 10. Pitch mask artifact

- `outputs/masks/<match>_pitch.png`: single-channel `uint8`, values {0, 255}, full clip resolution. 255 = pitch.
- Produced by Member 1 (segmentation).
- **Optionally** consumed by Member 2 (`yolo.pitch_filter: true`) to drop detections whose bottom-centre lies outside the pitch.
- If the file does not exist, consumers must run without the filter and set `notes: "pitch_filter=off"` in the metadata. Comparison tables report the filter state.

---

## 11. EDA statistics (`outputs/eda/stats.json`) — Member 1 → Member 4

```json
{
  "schema_version": "1.0",
  "created_utc": "2026-11-20T10:00:00Z",
  "matches": {
    "117093": {
      "split": "val", "image_width": 4096, "image_height": 1080, "fps": 25.0,
      "frames_annotated": 5976, "frames_decoded": 5976, "boxes": 131472, "ids": 22,
      "boxes_per_frame": {"min": 22, "max": 22, "mean": 22.0},
      "box_width_px": {"p05": 12.1, "median": 20.2, "p95": 33.0},
      "box_height_px": {"p05": 30.5, "median": 45.4, "p95": 70.2},
      "track_length_frames": {"min": 5976, "median": 5976, "max": 5976},
      "issues": {"degenerate_boxes": 0, "clipped_boxes": 0, "duplicate_rows": 0, "frames_not_expected": 0}
    }
  },
  "totals": {"matches": 7, "frames": 41932, "boxes": 922504},
  "paper_reference": {"frames": 59842, "boxes": 1316524, "median_box_wh_px": [25, 55]}
}
```

`outputs/eda/box_sample.csv` columns: `match,split,frame,x,y,w,h` (a documented subsample, e.g. every 7th row).

---

## 12. Phase 3 contracts (fixed now, so Phase 2 code stays compatible)

**PitchTrack table** (`frame,track_id,x_m,y_m,quality`):
- `x_m` ∈ [−52.5, 52.5] along the pitch length; `y_m` ∈ [−34, 34] across the width; origin at the pitch centre.
- The axis direction follows SoccerTrack's calibration convention (⚠️ to be verified by Member 4 before use).
- `quality` = 1 if the foot point (bottom-centre of the box) maps inside the pitch with valid calibration, else 0.

**Workload summary table:**
```
match,track_id,source,duration_s,distance_m,mean_speed_mps,peak_speed_mps,high_speed_distance_m,sprint_distance_m,n_accel,n_decel
```
- `source` ∈ {`gt`, `pred`}.
- Thresholds are in `configs/project.yaml` → `workload`: high speed ≥ 5.5 m/s, sprint ≥ 7.0 m/s, acceleration events |a| ≥ 3 m/s². These are conventional external-load thresholds, not physiological measures.

---

## 13. Configuration and path conventions

| File | Owner | Content |
|---|---|---|
| `configs/project.yaml` | Member 4 (split values frozen by the approved plan) | dataset version, split, seed, output roots, visualisation and workload settings |
| `configs/scratch.yaml` | Member 1 | segmentation, proposals, patches, HOG, SVM, CNN, scratch detector, IoU tracker |
| `configs/yolo.yaml` | Member 2 | weights key, tiling, inference, NMS, ByteTrack, fine-tuning (Phase 3) |
| `configs/evaluation.yaml` | Member 3 | framesets, score floor, IoU thresholds, AP settings, height bins, noisy-GT oracle, TrackEval |
| `configs/data_sources.example.yaml` | Member 4 | template with placeholders (committed) |
| `configs/data_sources.yaml` | each member, locally | absolute paths for this machine (**git-ignored**, never committed) |

**Rules:**
- Code reads configs only through `config.load_config()` and `config.load_data_sources()`.
- No hard-coded paths or parameters in modules.
- A member edits only their own config file. Changes to another member's config are requested through that owner (see `GIT_WORKFLOW.md`).
- `config.config_sha256()` hashes `project.yaml`, `scratch.yaml`, `yolo.yaml` and `evaluation.yaml`, but **not** `data_sources.yaml`, so frozen-test hashes are machine-independent.

---

## 14. Test-split access rule

- Any runner invoked with `--split test` must call `split.require_test_access(split, frozen_test)`. This raises unless `--frozen-test` is given.
- After a successful test run, the runner calls `split.log_test_access(...)`, which appends to `outputs/test_log.md`: date, member, method, `config_sha256`, model file hashes.
- Tuning code paths (threshold search, early stopping, grid search) must only ever load `train` or `val`.

---

## 15. Fixtures (`tests/fixtures/`, owned by Member 3, read-only for others)

A synthetic 3-frame sequence: image 640 × 360, 25 fps, 3 players.

| File | Content | Hand-computed expectations |
|---|---|---|
| `mot_raw_sample.txt` | Raw SoccerTrack format (0-based, −1 columns) for 3 frames × 3 IDs | Converts exactly to `tracks_gt.csv` |
| `mot_raw_dirty.txt` | Raw file with one duplicate row, one degenerate box, one box crossing the right border, one frame with 2 boxes | Cleaning report (with `expected_boxes_per_frame=3`; counts after de-duplication and degenerate-box removal): duplicate_rows 1, degenerate_boxes 1, clipped_boxes 1, frames_not_expected 1. Result: 11 boxes, 3 IDs (matches `999002` in `eda_stats_sample.json`) |
| `tracks_gt.csv` | GT Track table (9 rows) | — |
| `detections_sample.csv` | Detections for frames 0–2 (11 rows) | At IoU 0.5 and threshold 0.5: TP 7, FP 1, FN 2, P 0.8750, R 0.7778, F1 0.8235. All detections (threshold 0): TP 8, FP 3; all-point interpolated AP@0.5 = 0.8667 |
| `detections_sample.meta.json` | Matching metadata (frameset `fixture3`, 0–3 stride 1, 640 × 360) | — |
| `tracks_pred.csv` | Predicted tracks with one ID switch and two misses | MOTA = 0.6667 (FN 2, FP 0, IDSW 1, GT 9); IDF1 = 0.7500 (IDTP 6, IDFP 1, IDFN 3) |
| `tracks_pred.meta.json` | Matching metadata | — |
| `proposals_sample.csv` | 6 scored proposals | — |
| `patch_scores_sample.csv` | 8 patch scores | At threshold 0.5 (score ≥ threshold → positive): TP 3, FN 1, TN 3, FP 1 → Sensitivity 0.75, Specificity 0.75; ROC-AUC 0.9375 |
| `metrics_sample.json` | A valid E3 Metrics JSON for the fixture detections | Passes `validate_metrics`; all values are the hand-computed ones above except `ap50_95`, which is illustrative |
| `eda_stats_sample.json` | A valid EDA statistics file (2 synthetic matches) | — |
| `pitch_tracks_sample.csv` | Phase 3 PitchTrack example (one player moving at a constant 5 m/s) | Distance over 1 s (25 frames, 0.2 m per frame) = 4.8 m between first and last row; speed 5.0 m/s |
| `trackeval_example/` | Expected TrackEval export of `tracks_gt.csv` and `tracks_pred.csv` | Byte-for-byte target for `export_trackeval` |

Synthetic images, patches and videos are **generated in memory** by the test helpers in `tests/conftest.py`. No binary fixtures are committed.

The fixture frameset `fixture3` = `range(0, 3, 1)` is registered only for tests.
