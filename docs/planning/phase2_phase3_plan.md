# SoccerTrack v2 Player Detection, Tracking and External Workload — Phase 2 & Phase 3 Plan

| | |
|---|---|
| **Course** | SBES 426 — Medical Pattern Recognition |
| **Project** | Vision-Based Football Player Tracking and External Workload Analysis |
| **Document status** | Final planning document, Revision 3 (2026-11-19) with the Revision 3.1 team-preparation corrections below. Technical direction and data-reuse strategy provisionally approved; awaiting final approval before implementation |
| **Repository** | `D:\pattern_prj` |
| **Deadlines** | Phase 2: Wed 25 Nov 2026, 23:59 (online discussion Thu 26 Nov, 21:00) · Phase 3: Sat 19 Dec 2026 |
| **Workflow** | Claude Code: research/planning → ChatGPT + team: review/approval → Codex: implementation |
| **Companion** | [`transfer_manifest.md`](transfer_manifest.md): per-file paths, sizes, verification status, and reuse actions |

**Evidence legend:** ✅ verified directly (file contents, hashes, source text) · 📏 measured earlier by the Football Fatigue project on this machine (reference only, not re-run) · ⚠️ must be verified during implementation · 🧪 estimate.

> **Revision 3.1 (2026-11-19): corrections for the four-person team repository.** The research, data facts, split and evaluation design are unchanged. Superseding details live in `docs/team/`:
> 1. **Two separate trackers:** `tracker_iou.py` (IoU + Hungarian, Pipeline A, Member 1) and `tracker_bytetrack.py` (Ultralytics BYTETracker adapter, Pipeline B, Member 2). Both use the same Detection/Track contracts and the same evaluation.
> 2. **Configuration is split by owner:** `configs/project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml` (+ git-ignored `data_sources.yaml`) replace the single `config.yaml`. The frozen-test hash covers the four committed files.
> 3. **Added modules:** `config.py`, `contracts.py`, `compare.py`, `explain.py`. One runner per member replaces `scripts/run.py`.
> 4. **Metric domains:** bounded ratios (P, R, F1, AP, HOTA, IDF1, …) are stored as fractions in [0, 1] (HOTA = 1.0 means 100 %); MOTA ≤ 1 and can be negative; FP per frame and FPS are non-negative and can exceed 1; ID switches are integer counts (`docs/team/SHARED_CONTRACTS.md` §1).
> 5. **The Codex task list (Section 9) and the work packages (Section 10)** are superseded by the member milestones in `docs/team/MEMBER_*.md` and the integration order in `docs/team/INTEGRATION_PLAN.md`. Section 9.3 (T00 environment set-up) remains valid, except that its config file names follow item 2.
> 6. **Contracts:** `docs/team/SHARED_CONTRACTS.md` is the binding specification of all tables, metadata and Metrics JSON (it refines Section 4.2).

---

## Contents
1. Scope and principles
2. Authoritative data facts
3. Verified research findings
4. System architecture
5. Evaluation design and test protocol
6. Repository architecture
7. Phase 2 execution plan
8. Phase 3 strategy
9. Codex implementation handoff
10. Modularity for a four-person team
11. Risks, contingencies and open blockers
12. Decisions and approvals
13. Discussion defense notes
14. Sources

---

## 1. Scope and principles

**Goal.** Detect and track players in fixed panoramic football video (SoccerTrack v2). The long-term aim is to derive external workload indicators: trajectories, distance, speed, acceleration, and heatmaps.

**Principles**
1. **One extendable system.** Every stage communicates through a small set of shared file contracts (Section 4.2), so detectors, trackers, and analytics can be swapped or added without a rewrite.
2. **Educational and explainable first.** The from-scratch approach uses course techniques (segmentation, morphology, feature extraction, classical and CNN classification, Hungarian assignment). The pretrained approach is a reference.
3. **Strictly held-out testing.** All training, tuning, and failure-driven development uses training/validation matches only. The test matches are evaluated with a frozen configuration, once per phase.
4. **Reuse, do not duplicate.** Existing verified data and weights on `F:\Football-Fatigue-Project` are referenced read-only. Only the missing test matches are downloaded.
5. **Honest claims.** Workload indicators are external-load proxies. The dataset provides no physiological fatigue or injury ground truth, and none is claimed.

---

## 2. Authoritative data facts

### 2.1 Data sources

| Role | Matches (paper label) | Location | Status |
|---|---|---|---|
| Train | 118575 (M3), 118576 (M4), 118577 (M5), 118578 (M6), 128058 (M8) | `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\{<id>.txt, clips\<id>.mp4}` | ✅ local, verified, read-only |
| Train (optional) | 117092 (M1) | `F:\Football-Fatigue-Project\data\raw\soccertrack_v2\mot\{117092.txt, 117092.mp4}` | ✅ local. Different camera geometry and a known stitch seam; 588 MB clip |
| Validation | 117093 (M2) | txt: `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\117093.txt`; clip: `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4` | ✅ local, verified |
| **Test (held out)** | **128057 (M7), 132831 (M9)**, the official SoccerTrack Challenge 2025 test pair | to be downloaded to `D:\pattern_prj\data\raw\soccertrack-v2\mot\` | ❌ missing; **≈ 0.91 GB** |
| Validation (optional) | 132877 (M10) | to be downloaded | ❌ optional; ≈ 0.31 GB |
| Calibration (Phase 3) | keypoints/intrinsics for 117092, 117093, 118575, 118576 (+ homography for 117093) | `F:\…\data\external\soccertrack-v2\raw\<id>\` | ✅ local |
| Calibration (Phase 3, test) | `raw/{128057,132831}/` small files + GitHub `data_corrections/132831_keypoints.json` | to be downloaded | ≈ 15 KB |
| Weights (Approach B) | COCO YOLOv8n (primary), YOLOv8s (Phase 3 fine-tuning base) | `F:\Football-Fatigue-Project\models\yolov8n.pt`, `yolov8s.pt` | ✅ local, read-only |

All local SoccerTrack files are HF release **v1.2** (commit `325bd4c57ee8e135ab652f3921d274526ada496b`). Annotation files are byte-identical to the HF release (git-blob SHA-1). Six clips are SHA-256-verified; 117092 is size-verified only. Details: `transfer_manifest.md`.

**Split rationale.** Test = official M7 + M9; validation = M2. This follows the HF README v1.2 and paper protocol, except that the optional M10 is not included by default. The split is at match level: the two halves of a match share teams, camera, weather, and lighting, so frame-level or within-match splits would leak.

### 2.2 Verified properties of the MOT annotation files
Verified on all seven local files:

| Property | Value | Consequence |
|---|---|---|
| Columns | 10, comma-separated, no header: `frame,id,x,y,w,h,conf,class,vis,unused` | Fixed-schema parser |
| Frame index | **0-indexed** (0 … N−1). The official `docs/format-mot.md` describes 1-indexing, but the released files do not follow it | Internal representation is 0-based; the TrackEval adapter writes `frame+1` |
| `conf`, `class`, `vis` | **−1 in every row** | No ignore regions, no visibility labels. TrackEval export must write `conf=1, class=1, vis=1` |
| Boxes per frame | exactly 22; 22 IDs per clip | Only the 22 players are labelled. Referees, staff, and spectators are not, so detecting them produces explainable false positives |
| Frames per clip | 6,000 (117092, 118575, 118577, 118578); 5,976 (117093); 5,975 (118576); 5,981 (128058) | "Missing data" = slightly truncated clips; frame count must equal the decoded video frame count ⚠️ |
| Box size | median ≈ 20 × 45 px (sample of the 7 files) vs 25 × 55 px in the paper | Players are tiny; whole-frame downscaling is not allowed |
| Totals (7 files) | 922,504 boxes / 41,932 frames. The paper's totals (1,316,524 / 59,842) imply 22 × 17,910 for the 3 missing clips | Full reproduction check after download |

Additional facts (📏, from the earlier project, to re-check ⚠️):
- Clip 117093 is 4096 × 1080 at 25 fps with 5,976 decoded frames. The MP4 container `start_time` is 0.979 s, so frames must be decoded sequentially by index, never sought by timestamp.
- Keypoint-based calibration on 117093 had a held-out landmark error of median 2.61 m (p95 4.44 m).

### 2.3 Prior Football Fatigue results (reference only)
Earlier work on validation match 117093:
- Tiled COCO YOLOv8n: P 0.846, R 0.780, F1 0.812 (1,000 frames, confidence ≥ 0.40).
- Full-clip ByteTrack IDF1 0.24–0.27.
- About 90 % of false positives were unlabelled people.

These numbers are **not** this project's experimental results. They serve only as a sanity range for our validation evaluation and as failure-mode hypotheses.

### 2.4 Excluded resources
FIFA World Cup 2022 PFF data (no video, no boxes, incompatible coordinate and sampling conventions, licence unverified), Wadi Degla videos (no annotations, consent undocumented), the earlier project's generated outputs (~35 GB), the third-party football detector (training data provenance unclear; optional extra row only), and the GSR JSON and full-match videos (not needed; GSR boxes are synthetic, generated from pitch positions).

---

## 3. Verified research findings

### 3.1 Official repository — https://github.com/AtomScott/SoccerTrack-v2
- Code MIT; data CC BY 4.0.
- The baseline starter kits are "scaffolding". **No pretrained weights** are provided, and the MOT leaderboard JSON is empty.
- The MOT starter kit (YOLOv8-s + ByteTrack) expects full-half videos and a different folder layout, so it is not runnable on the HF MOT subset as-is.
- The MOT evaluator (`src/evaluation/mot_hota.py`) wraps TrackEval `MotChallenge2DBox`, with `SKIP_SPLIT_FOL=True` and a per-sequence `seqinfo.ini`.
- Test match 132831 had two transposed calibration keypoints; the corrected file is `data_corrections/132831_keypoints.json`.

### 3.2 Hugging Face dataset — https://huggingface.co/datasets/atomscott/soccertrack-v2
- Gated (auto-approval after accepting terms), ≈ 144.8 GB in total, release v1.2.
- Official split: test M7 + M9, validation M2 + M10, train the other six matches.
- GSR boxes are generated from pitch positions (not human-drawn), so they are unsuitable as detector ground truth.
- Pitch: 105 × 68 m, origin at centre, positions quantised to 1.05 m × 0.68 m.

### 3.3 Paper — https://arxiv.org/abs/2508.01802 (v2)
- MOT subset: one 4-minute clip per match, 59,842 frames, 1,316,524 boxes, 22 players per frame, median box 25 × 55 px.
- GSR baseline GS-HOTA: 12.25 over all 20 halves, 10.96 over the test halves, 31.35 for independent 30-s runs. This is a pitch-space, attribute-aware, 45-minute task.
- BAS mAP@1s: 0.222 / 0.599.
- The paper states that the reference MOT results are on the SoccerTrack Challenge leaderboard, not in the paper.

### 3.4 SoccerTrack Challenge 2025 (MMSports '25)
- Ranked by HOTA averaged over the two test videos (M7 + M9). Top-5 HOTA: 0.60, 0.59, 0.59, 0.59, 0.59.
- Winner GTATrack (https://arxiv.org/abs/2602.00484): fine-tuned YOLOv11x (200 epochs, RTX 3090) + Deep-EIoU + OSNet ReID + GTA-Link.
- Its own-validation ByteTrack baseline: HOTA 0.42 (not the test set).

### 3.5 Comparability

| Reference | Comparable with our work? | Use |
|---|---|---|
| Paper dataset statistics | ✅ reproducible from MOT files | Reproduce; report the median-box difference |
| Challenge HOTA 0.59–0.60 | ✅ approximately: same clips and test matches; requires full 4-minute test clips; exact challenge configuration not fully documented | Upper-bound reference for tracking |
| GTATrack ByteTrack 0.42 | ⚠️ their validation split | Indicative only |
| Prior Football Fatigue numbers | ⚠️ validation match, different pipeline | Sanity check only |
| Paper GS-HOTA / LocA | ❌ different task, space, attributes, and sequence length | Explain non-comparability |
| Paper BAS mAP | ❌ different task | Context only |

---

## 4. System architecture

### 4.1 Overview

```
      configs/{project,scratch,yolo,evaluation}.yaml + configs/data_sources.yaml
                                        │
                              ┌─────────▼─────────┐
                              │  data layer        │  catalog · MOT loader · video reader · split
                              └───┬───────────┬────┘
               frames (BGR, 0-based index)    │ ground truth (Track table)
          ┌───────────────────────┴──┐        │
          ▼                          ▼        │
 ┌──────────────── APPROACH A ─────────────┐  │  ┌──────── APPROACH B ────────┐
 │ segmentation: pitch mask, background,    │  │  │ tiling: native-resolution  │
 │   foreground mask, morphology            │  │  │   overlapping tiles        │
 │ proposals: connected components,         │  │  │ YOLOv8n (COCO "person")    │
 │   perspective size filter → Proposal tbl │  │  │ tile→frame coords, NMS     │
 │ features: HOG (+ colour)                 │  │  └─────────────┬──────────────┘
 │ classifier: HOG+SVM | Small CNN          │  │                │
 │ NMS → Detection table                    │  │                │
 └──────────────────────┬───────────────────┘  │                │
                        └──────────► Detection table ◄──────────┘   (shared contract)
                                         │
                                         ▼
                         tracking: IoU + Hungarian (+ Kalman, later)
                                         │
                                         ▼
                                    Track table  ◄─────── ground truth uses the same contract
                     ┌───────────────────┼─────────────────────────────┐
                     ▼                   ▼                             ▼
            evaluation (E1–E4)    pitch mapping (Phase 3)       visualisation
                                         │                     (overlays, figures)
                                         ▼
                              PitchTrack table → workload metrics, heatmaps (Phase 3)
```

### 4.2 Shared data contracts
All tables are CSV files (small enough; human-readable) with fixed columns, validated by one schema function per contract.

| Contract | Columns | Units / conventions | Produced by | Consumed by |
|---|---|---|---|---|
| **Track table** (GT and predictions) | `frame, track_id, x, y, w, h, score` | `frame` 0-based int; box top-left `x, y`, width/height in pixels of the original frame; `score` ∈ [0,1] (GT = 1) | MOT loader, tracker | evaluation, pitch mapping, visualisation |
| **Detection table** | `frame, x, y, w, h, score` | same box convention | Approach A, Approach B, GT-oracle adapter | tracker, detection evaluation |
| **Proposal table** | `frame, x, y, w, h, area, fill_ratio` | same box convention | proposals module | patch extraction, Approach A classifier, E2 |
| **Patch set** (`.npz`) | `X` uint8 [N,64,32,3], `y` {0,1}, `match`, `frame`, `kind` (gt / random / partial / proposal) | RGB, 64 high × 32 wide | patches module | classifiers, E1 |
| **Metrics JSON** | `{run_id, split, match, method, metrics{…}, config_sha256, data_sha256, created}` | — | evaluation | notebook, report tables |
| **PitchTrack table** (Phase 3) | `frame, track_id, x_m, y_m, quality` | metres, origin at pitch centre, SoccerTrack axis convention | pitch module | workload module |
| **TrackEval export** (adapter only) | MOTChallenge `frame+1,id,x,y,w,h,1,1,1` + `seqinfo.ini` | 1-based, per evaluator requirements | tracking evaluation adapter | TrackEval 1.3.0 |

Because every detector emits the same Detection table, the tracker, the evaluators, and the Phase 3 analytics are independent of the detector. Ground truth uses the same Track table, so tracking can be evaluated on GT boxes (the oracle) before any detector exists.

### 4.3 Module boundaries and interfaces

| Module | Responsibility | Key interface (behavioural contract, not code) | Depends on |
|---|---|---|---|
| `data_io` | Path catalog, MOT parsing/validation, sequential video decoding, frame sampling, Track-table I/O | `load_tracks(match) → Track table`; `iter_frames(match, frame_ids) → (idx, image)` | config |
| `split` | Match-level split, leakage guards | `matches(split)`; assertions that train/val/test are disjoint and that the test set is not read during tuning | config |
| `segmentation` | Pitch mask (HSV / keypoint polygon / manual polygon), median background, foreground mask, morphology | `pitch_mask(match)`, `background(match)`, `foreground(image, bg, mask) → binary` | data_io |
| `proposals` | Connected components, perspective size model h(y) (fitted on train GT), blob splitting | `propose(binary, frame) → Proposal table` | segmentation |
| `patches` | Positive/negative patch sampling (1:3), crops resized to 64×32 | `build(split) → Patch set` | data_io, proposals |
| `features` | HOG (756-d for 64×32), colour histograms | `hog(patches) → array` | — |
| `classifiers` | HOG+Linear SVM; Small CNN; fit/predict/save/load | `fit(X, y)`, `predict_proba(patches) → array` | features |
| `detector_scratch` | Approach A end-to-end | `detect(match, frames) → Detection table` | segmentation, proposals, classifiers |
| `detector_yolo` | Approach B: tiling, YOLOv8n inference, tile merge, NMS | `detect(match, frames) → Detection table` | data_io |
| `tracker` | IoU + Hungarian (Kalman flag later) | `run(Detection table) → Track table` | — |
| `metrics` | IoU, NMS, matching, P/R/F1/AP, confusion-matrix helpers | pure functions | — |
| `evaluate` | E1–E4 drivers; TrackEval adapter | `eval_detection(gt, det, frames) → Metrics JSON`, `eval_tracking(gt, tracks) → Metrics JSON` | metrics, data_io |
| `pitch` (Phase 3) | Homography / fisheye model from keypoints; foot point → metres | `to_pitch(Track table, match) → PitchTrack table` | data_io |
| `workload` (Phase 3) | Smoothing, distance, speed, acceleration, speed bands, heatmaps | `summarise(PitchTrack table) → per-player table` | — |
| `visualize` | Overlays, stage figures, plots | functions returning figures/images | contracts only |

### 4.4 Approach A — from scratch
1. **Segmentation** (label-free, per clip):
   - Pitch mask: HSV green threshold on the median background, then largest component and hole filling. Fallback: polygon from calibration keypoints (⚠️ pixel-space check) or a manual polygon.
   - Background: per-pixel median of ~60 frames sampled across the clip.
   - Foreground: |blur(frame) − background| > τ inside the pitch mask, then median filter, opening, closing, and a small vertical dilation.
2. **Region proposals:** connected components → boxes, kept if their size is consistent with the expected player height at that image row, h(y), fitted on training GT. Oversized blobs are split (merged players).
3. **Classification:** each proposal is cropped with 10 % context and resized to 64×32.
   - (a) HOG (8×8 cells, 2×2 blocks, 9 bins → 756-d) [+ HSV colour histogram] → Linear SVM with balanced class weights.
   - (b) Small CNN: 3 × (Conv-BN-ReLU-MaxPool) → global average pooling → FC; about 25k parameters; weighted cross-entropy.
4. **Detection:** score threshold (chosen on validation) → NMS → Detection table.
5. **Minimum working baseline (MWB-0):** steps 1–2 plus a rule-based verifier (size, aspect, fill ratio). MWB-1 replaces the verifier with HOG+SVM.

### 4.5 Approach B — pretrained
- Native-resolution overlapping tiles. Starting point: 1280-px-wide tiles with ≥ 64 px overlap; tile size and overlap are tuned on validation.
- COCO-pretrained YOLOv8n, class "person".
- Tile → frame coordinates, cross-tile NMS, pitch-mask filter (shared with Approach A).
- Output: the same Detection table.
- Phase 3: fine-tune YOLOv8n/s on training tiles (transfer learning).

### 4.6 Tracking
- **Two separate trackers (Revision 3.1):**
  - Pipeline A: from-scratch IoU + Hungarian tracker (`tracker_iou.py`). Cost = 1 − IoU; assignment with `scipy.optimize.linear_sum_assignment`; IoU gate; track confirmation after `n_init` hits and deletion after `max_age` misses.
  - Pipeline B: ByteTrack via the Ultralytics BYTETracker (`tracker_bytetrack.py`).
  - Parameters of both are tuned on validation. Both trackers are also run on GT boxes (oracle).
- **Validation order:**
  1. GT boxes (oracle)
  2. GT boxes with simulated detector noise (drop 10 %, jitter)
  3. YOLO detections
  4. Approach A detections
- Phase 2: should-have. Phase 3: core capability, with a Kalman motion model, two-stage (high/low score) association, and short-gap stitching. ByteTrack (ultralytics) is the reference.

### 4.7 Why Phase 3 requires no rewrite

| Phase 3 feature | Attaches to | Change needed |
|---|---|---|
| YOLO fine-tuning | `detector_yolo` (weights path in config) | New weights only; same Detection table |
| From-scratch improvements (hard negatives, blob splitting, CNN tuning) | `patches`, `proposals`, `classifiers` | Internal to Approach A; same Detection table |
| Tracker improvements (Kalman, two-stage, stitching; ByteTrack tuning) | `tracker_iou` / `tracker_bytetrack` (config flags) | Same Detection-in / Track-out contract |
| Explainability (Grad-CAM, HOG/SVM maps, feature importance) | `classifiers` + `visualize` | Read-only use of trained models and patch sets |
| Full-clip tracking evaluation | `evaluate` | Longer frame ranges only |
| Pitch mapping | new `pitch` module consuming the Track table | Additive |
| Workload metrics, heatmaps, tactical board | new `workload` module consuming PitchTrack + `visualize` | Additive; works identically on GT tracks and predicted tracks |
| Team identification (optional) | additive module consuming the Track table + frames | Additive |

---

## 5. Evaluation design and test protocol

### 5.1 Evaluations

| ID | Question | Data | Metrics |
|---|---|---|---|
| **E1 Patch classification** | Does the classifier separate player vs. background? | Patch sets per split. Positives = GT crops; negatives = random pitch windows (IoU < 0.1), partial windows (0.1–0.4), unmatched proposals; fixed 1:3 ratio | Confusion matrix, Sensitivity, Specificity, Precision, F1, ROC-AUC, PR-AUC |
| **E2 Proposal classification** | How does it perform on the real (imbalanced) candidate distribution? | All Approach A proposals on evaluation frames; label = IoU ≥ 0.5 with GT | Same as E1 + PR curve |
| **E3 End-to-end detection** | How many players are found, and where? | Every 10th frame of each evaluation clip (≈ 600 frames, ≈ 13k GT boxes per clip) | P/R/F1 at IoU 0.5 (and 0.3), AP@0.5, AP@[.5:.95], recall per box-height bin, FP per frame, proposal recall (Approach A upper bound), frames per second |
| **E4 Tracking** | Do IDs remain consistent? | Contiguous 25-fps window: 60 s in Phase 2; full clips in Phase 3 | HOTA, DetA, AssA, LocA, IDF1, MOTA, IDSW (TrackEval 1.3.0) |

Fairness: Approaches A and B use the same frames, GT, IoU thresholds, evaluation code, pitch mask, and hardware. Results are reported per match and pooled. COCO pretraining means A vs B compares *systems*, not equal-data architectures. Phase 3 fine-tuning uses only the training split.

Sanity checks (unit tests):
- GT as prediction gives AP = 1 and HOTA = 1.0 (100 %).
- Our own AP agrees with `pycocotools` within 0.01.
- YOLOv8n on validation lands near the earlier reference range; a large deviation triggers a bug review.

### 5.2 Held-out test protocol
1. All decisions (thresholds, kernels, τ, `C`, epochs, tracker parameters, tile settings) are made on validation match 117093.
2. Failure analysis that motivates improvements uses training/validation data only.
3. Before a test run, the configuration is frozen. The SHA-256 of the committed config files, model file hashes, and data hashes are recorded in `outputs/test_log.md`.
4. One test run per frozen system per phase (Phase 2 and Phase 3 final). No change is made in response to test results. A bug that invalidates a run is documented and the run repeated once.
5. Test patch sets are built only inside the frozen test run.
6. Leakage guards (tested): disjoint match sets; normalisation statistics, size model, and samplers read training data only; per-clip background and pitch mask are label-free.

---

## 6. Repository architecture

### 6.1 Final folder tree

The complete tree with file owners is in `docs/team/INTEGRATION_PLAN.md` (Section 2). Summary:

```
D:\pattern_prj
├── README.md  requirements.txt  pyproject.toml  .gitignore
├── configs/
│   ├── project.yaml                  # dataset version, split, seed, visualisation, workload (Member 4)
│   ├── scratch.yaml                  # Approach A + IoU tracker (Member 1)
│   ├── yolo.yaml                     # Approach B + ByteTrack (Member 2)
│   ├── evaluation.yaml               # framesets, score floor, IoU thresholds, TrackEval (Member 3)
│   ├── data_sources.example.yaml     # committed template of machine-specific paths
│   └── data_sources.yaml             # this machine's absolute paths (git-ignored)
├── src/soccertrack/
│   ├── __init__.py  config.py                                   # Member 4
│   ├── contracts.py  metrics.py  evaluate.py  compare.py  explain.py               # Member 3
│   ├── data_io.py  split.py  segmentation.py  proposals.py  patches.py  features.py
│   │   classifiers.py  detector_scratch.py  tracker_iou.py                         # Member 1
│   ├── detector_yolo.py  tracker_bytetrack.py                                      # Member 2
│   └── visualize.py  pitch.py  workload.py                                         # Member 4
├── scripts/  run_scratch.py  run_yolo.py  run_evaluation.py  run_visualization.py  check_data.py  download_test_matches.py
├── notebooks/  phase2_baseline.ipynb  phase3_final.ipynb
├── tests/  conftest.py  fixtures/  test_*.py (one or two files per module)
├── docs/  phase1/ (unchanged)  planning/  team/ (contracts, guides, prompts, workflow)  phase2/  phase3/
├── data/      (git-ignored) raw/soccertrack-v2/ (test download)  manifests/  processed/
└── outputs/   (git-ignored) detections/ tracks/ proposals/ scores/ masks/ eda/ metrics/ tables/ figures/ trackeval/ test_log.md
```

### 6.2 Responsibilities

| Area | Responsibility |
|---|---|
| `configs/` | Single source of parameters (four owner-specific YAML files) and of machine paths (`data_sources.yaml`). No parameter is hard-coded in modules |
| `src/soccertrack/` | All reusable logic, one module per responsibility (Section 4.3) |
| `scripts/` | Thin entry points only (data checks, download, stage runner) |
| `notebooks/` | Narrative and figures for submission; they call `src/` and read saved outputs; they contain no core logic |
| `tests/` | Unit and contract tests on synthetic fixtures; no network, no large data |
| `docs/` | Planning, reports, slides, small curated figures |
| `data/`, `outputs/` | Local, regenerable artefacts; never committed |

### 6.3 Configuration and dataset paths
- `data_sources.yaml` maps each match to its `mot_txt` and `clip` paths, and maps `weights.yolov8n` / `weights.yolov8s` and `calibration.<match>` to their files.
- On this machine, train/validation entries point read-only to `F:\Football-Fatigue-Project\…`; test entries point to `D:\pattern_prj\data\raw\soccertrack-v2\…`.
- Team members on other machines fill in their own copy from `data_sources.example.yaml`.
- `project.yaml` holds the split and seed; `scratch.yaml`, `yolo.yaml` and `evaluation.yaml` hold the method and evaluation parameters (Revision 3.1).
- No module writes to any path listed as read-only. `check_data.py` verifies existence, byte size, hash (from the manifest), resolution, fps, and that the decoded frame count equals the annotated frame count.

### 6.4 Notebook and report structure
**`phase2_baseline.ipynb`** — "fast mode" reads saved outputs; "full mode" re-runs stages.
1. Problem and dataset
2. EDA and data quality
3. Preprocessing
4. Split and leakage controls
5. Segmentation and proposals
6. Patch dataset and class imbalance
7. HOG+SVM
8. Small CNN
9. Detection: Approach A vs Approach B
10. Tracking (if completed)
11. Frozen test results
12. Comparison with paper and Challenge
13. Limitations and Phase 3 plan

**Phase 2 report** (mirrors the assignment): preprocessing pipeline and data split → baseline architectures and training settings → evaluation results and comparison with the reference paper → limitations and Phase 3 plan.

**Slides:** about 10–12 slides following the same order.

**Phase 3 notebook/report:** adds improvements, explainability, failure analysis, final test evaluation, pitch mapping and workload, limitations, and future work.

### 6.5 Tests
- One test file per module, using synthetic fixtures:
  - MOT parsing (0-based frames, −1 columns, degenerate boxes)
  - TrackEval export (1-based frames, `conf/class/vis = 1`, `seqinfo.ini`)
  - split disjointness and leakage guards
  - segmentation/proposals on synthetic green images
  - patch sampling IoU bounds
  - HOG dimensionality (756) and agreement with OpenCV's `HOGDescriptor` within tolerance
  - CNN forward shape and one-batch overfit
  - IoU/NMS/AP on hand-computed cases
  - tracker identity on synthetic crossing objects
  - GT-as-prediction → AP = 1, HOTA = 1.0
  - Phase 3: homography on synthetic points; speed of a constant-velocity track
- Contract tests validate every table against its schema (Section 4.2), so modules can be developed in parallel against fixtures.

### 6.6 Git-ignore policy and reproducibility
- **Ignored:** `data/`, `outputs/`, `configs/data_sources.yaml`, `.venv/`, caches, `*.mp4`, `*.pt`, `*.pth`, `*.npz`, `*.npy`, `.env*`, notebook checkpoints. The existing `.gitignore` already covers most of these; `configs/data_sources.yaml`, `*.npz`, and `*.npy` are to be added.
- **Committed:** code, configs (except local paths), tests and fixtures, docs, small curated report figures under `docs/phase2|3/`, and notebooks with outputs cleared.
- **Pinned environment** (known-good on this machine 📏): numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.1.1, matplotlib 3.11.2, opencv-python 5.0.0.93, torch 2.9.1+cu126, torchvision 0.24.1+cu126, ultralytics 8.4.182, trackeval 1.3.0, lap 0.5.13, pycocotools 2.0.11, pyyaml 6.0.3, tqdm 4.70.1, pytest 9.1.1. Add `ipykernel` / `nbconvert`; their versions are recorded at install time.
- **Run provenance:** every metrics JSON records `run_id`, git commit (if any), the config SHA-256, input data and model hashes, package versions, and seed.
- **Determinism:** fixed seeds for Python, NumPy, and PyTorch; cuDNN deterministic mode where supported; decoding by frame index.
- **Data provenance:** HF release v1.2; local files verified against the pinned manifest; SoccerTrack (CC BY 4.0) and Ultralytics (AGPL-3.0) licences acknowledged in the README and report.

---

## 7. Phase 2 execution plan

### 7.1 Minimum Phase 2 submission path (must-have)
Each item is required. Together they form an independently submittable Phase 2.
1. Environment, data catalog, and verification of all local files; download and verification of the two test matches.
2. EDA and data-quality report (all 9 or 10 MOT files): reproduced statistics, 0-based and −1 findings, box-size distribution, frames per clip.
3. Match-level split, leakage guards, and test-log template.
4. Approach A segmentation + proposals with proposal recall on validation → **MWB-0** (rule-based verifier, E3 on validation).
5. Patch sets (train/val) with imbalance handling → **HOG+SVM (MWB-1)** and **Small CNN**, E1 on validation.
6. Approach A end-to-end detector, E2 + E3 on validation.
7. Approach B YOLOv8n tiled detector, E3 on validation.
8. Configuration freeze → **one test run**: E1, E2, E3 for A and B on 128057 and 132831.
9. Comparison tables (A vs B; reproduced dataset statistics; Challenge reference with caveats; GS-HOTA non-comparability).
10. `phase2_baseline.ipynb`, report, slides, limitations, Phase 3 plan.

### 7.2 Extensions
- **Should-have:** IoU + Hungarian tracker with E4 on validation (oracle GT, noisy GT, YOLO detections) and on a 60-s window of each test clip in the frozen run; a short A-vs-B overlay video for the presentation.
- **Optional:** tracker on Approach A detections; ByteTrack row; Kalman flag; optional validation match 132877; ~10 hand-annotated masks for a segmentation Dice score; third-party football detector as a labelled extra row.

### 7.3 Schedule

| Day | Work (task IDs from Section 9) | Milestone |
|---|---|---|
| Thu 19 | Approval; accept HF terms and create a token; user starts the test download (T02) | — |
| Fri 20 | T00, T01, T03, T04, T05, T06 | **MWB-0 on validation** |
| Sat 21 | T07, T08, T09 | **MWB-1 + Small CNN (E1)** |
| Sun 22 | T10, T11; T13 (should-have) | Validation results complete; configuration candidate |
| Mon 23 | T12 frozen test run; T14 tables. **Cut line 18:00:** drop unfinished should-haves | Frozen results |
| Tue 24 | T15 notebook; report; slides; rehearsal | Drafts |
| Wed 25 | Buffer; final checks; submit | Submission |

### 7.4 Compute expectations (GTX 1650 Ti 4 GB, i5-10300H, 32 GB RAM)

| Step | Cost |
|---|---|
| Approach A segmentation and proposals (CPU, 0.5 scale) | ~5–15 frames/s 🧪 |
| Patch sets (~60–80k patches, ≈ 0.5 GB) | ~10–20 min 🧪 |
| HOG+SVM training | minutes 🧪 |
| Small CNN training (~15 epochs) | ~5–15 min on GPU 🧪 |
| YOLOv8n tiled inference | 0.12–0.16 s/frame 📏 → E3 on 1,200 frames ≈ 3 min; full clip ≈ 13–16 min |
| Tracker + TrackEval | seconds to a minute 🧪 |
| Phase 3 YOLO fine-tuning (n/s, 640 native tiles) | ~1–3 h 🧪; Colab/Kaggle as fallback |

---

## 8. Phase 3 strategy (26 Nov → 19 Dec)

1. **Failure analysis on training/validation data:**
   - Errors by box height, image region, and crowding.
   - False positives: unlabelled people vs. shadows/lines vs. duplicates.
   - Misses: merged blobs, small/far players, goalkeepers.
   - ID switches at crossings.
   - Output: a ranked list that justifies the improvements.
2. **Improvements** (at least one required; chosen by evidence):
   - **I-1 Transfer learning** (expected primary): fine-tune YOLOv8n/s from the local COCO weights on training tiles, with early stopping on validation.
   - **I-2 Approach A:** hard-negative mining, size-guided splitting of merged blobs or sliding windows, a referee/staff colour-outlier cue, CNN hyperparameter search.
   - **I-3 Tracking:** Kalman motion, two-stage association, small-box-aware IoU gating, short-gap stitching; compared with ByteTrack.
3. **Final frozen test run** on the full test clips. Same metrics as Phase 2; Phase 2 → Phase 3 Δ table; comparison with the Challenge (HOTA 0.59–0.60, approximately comparable).
4. **Explainability:** Grad-CAM on the Small CNN (correct vs. misclassified patches), HOG glyphs and SVM weight maps, permutation importance of feature groups (HOG / colour / shape). EigenCAM on YOLO is optional.
5. **Pitch mapping and external workload:**
   - Foot point → metres using the calibration keypoints with the fisheye intrinsics (corrected keypoints for 132831). Report leave-one-out landmark error.
   - Smoothing, distance, speed, acceleration, time in conventional speed bands, heatmaps.
   - Validated by comparing metrics from GT tracks vs. predicted tracks through the same mapping.
   - Reported with calibration uncertainty (~2–3 m), on 4-minute clips. No physiological claims.
6. **Bonus (after 1–5):** A-vs-B overlay video, 2D tactical board, per-player workload cards, simple static dashboard, optional team-colour clustering evaluated against ~44 manual track labels.

---

## 9. Codex implementation handoff

> Revision 3.1: the task IDs below are superseded by the member milestones in `docs/team/MEMBER_*.md`; Section 9.3 remains the environment set-up reference.

**Global rules for Codex:**
- Work only in `D:\pattern_prj`. Treat every `F:\` path as read-only.
- Never modify `docs/phase1/`.
- Use pinned versions; install only with approval.
- Fixed seeds; no test-split access outside T12/T22.
- Every module gets docstrings and pytest tests on synthetic fixtures.
- Commits only when the team asks.

### 9.1 Phase 2 tasks

| ID | Depends on | Task | Outputs | Tests | Acceptance criteria |
|---|---|---|---|---|---|
| **T00** | approval | Project setup (see 9.3) | skeleton, configs, `requirements.txt`, `.gitignore` update, `.venv` | import/CUDA smoke test; config schema test | Section 9.3 checklist passes |
| **T01** | T00 | Data catalog + verification | `data_io` catalog functions; `scripts/check_data.py`; `outputs/data_check.json` | path resolution, size/hash checks, missing-file reporting (temp files) | All 7 local matches pass: hash/size, resolution, fps, decoded frames = annotated frames |
| **T02** | T00 | Test-match download script (run by the user) | `scripts/download_test_matches.py` (HF revision v1.2, `HF_HOME` and temp directories on D:) | include-pattern builder (no network) | After the user runs it, `check_data.py` passes for 128057 and 132831 against the pinned manifest |
| **T03** | T01 | MOT loader + EDA + quality report | `data_io` loaders; `outputs/eda/*` | 0-based parsing, −1 columns, degenerate/duplicate rows, Track-table schema | Per-match stats; comparison with the paper's totals (exact once all available); 3 overlay frames per clip show alignment |
| **T04** | T03 | Split + leakage guards + test log | `split.py`; `outputs/test_log.md` template | disjointness; training-only readers | Split table produced |
| **T05** | T04 | Segmentation + proposals (with fallbacks) | `segmentation.py`, `proposals.py`; stage figures; proposal recall on validation | synthetic green image → expected boxes; morphology; polygon masks | Parameters chosen on validation; fallback selectable in config |
| **T06** | T05 | Metrics + detection evaluation + MWB-0 | `metrics.py`, `evaluate.py` (E3); MWB-0 Detection table and E3 on validation | IoU/NMS/AP hand cases; GT-as-prediction AP = 1; agreement with `pycocotools` ≤ 0.01 | MWB-0 E3 results on validation |
| **T07** | T05 | Patch sets (train/val) | `patches.py`; `data/processed/patches_{train,val}.npz`; class-count table | IoU bounds; split membership; shape 64×32×3 | 1:3 ratio documented; no test patches |
| **T08** | T07 | HOG + Linear SVM (MWB-1) | `features.py`, `classifiers.py`; model; E1 on validation | HOG = 756-d; matches `cv2.HOGDescriptor` within tolerance; determinism | `C` and threshold chosen on validation; confusion matrix, ROC, PR |
| **T09** | T07 | Small CNN | `classifiers.py` (CNN); weights; curves; E1 on validation | forward shape; one-batch overfit; < 100k parameters | Early stopping on validation |
| **T10** | T06, T08, T09 | Approach A end-to-end detector | `detector_scratch.py`; Detection tables; E2 + E3 on validation | NMS and threshold logic | Beats MWB-0 on validation F1, or the shortfall is explained |
| **T11** | T06 | Approach B tiled YOLOv8n | `detector_yolo.py`; Detection tables; E3 + FPS on validation | tile↔frame round trip; cross-tile duplicate merge | Validation P/R within the reference range, else investigated |
| **T12** | T02, T10, T11 (T13) | **Frozen Phase 2 test run** | test patch sets; E1–E3 (+E4) for 128057, 132831; `test_log.md` entry | — | Exactly one logged run per system; per-match and pooled metrics |
| **T13** *(should)* | T06 | Tracker + TrackEval adapter | `tracker.py`; `evaluate.py` (E4); Track tables; E4 on validation (oracle → noisy → YOLO) | crossing objects keep IDs; birth/death; export is 1-based with `conf/class/vis = 1` and `seqinfo.ini`; GT-as-prediction HOTA = 1.0 | Oracle and YOLO E4 on validation |
| **T14** | T12 | Comparison tables + figures | `outputs/metrics`, `outputs/figures`, curated copies under `docs/phase2/` | schema validation of Metrics JSON | Includes the comparability table (Section 3.5) |
| **T15** | T14 | Phase 2 notebook | `notebooks/phase2_baseline.ipynb` | executes in fast mode (`nbconvert --execute`) | Structure of Section 6.4; outputs cleared before commit |

### 9.2 Phase 3 tasks

| ID | Depends on | Task | Acceptance criteria |
|---|---|---|---|
| T16 | T15 | Failure analysis (training/validation only) | Ranked failure list with stratified tables and galleries |
| T17 | T16 | YOLO fine-tuning (I-1) | Validation improvement documented before testing |
| T18 | T16 | Approach A improvements (I-2) | Validation Δ vs Phase 2 |
| T19 | T13, T16 | Tracker improvements (I-3), ByteTrack reference, full-clip E4 on validation | Validation Δ HOTA/IDF1/IDSW |
| T20 | T08, T09 | Explainability | Grad-CAM, HOG/SVM maps, permutation importance; correct and error cases |
| T21 | T13 | Pitch mapping + workload | Leave-one-out landmark error in metres; GT-track vs predicted-track workload error |
| T22 | T17–T21 | **Final frozen test run** | Second and last logged test access; full test clips |
| T23 | T22 | Phase 3 notebook, report, presentation; optional bonus visualisations | Reproducible from config + data catalog |

### 9.3 First Codex task (T00): exact specification
**Goal:** a runnable, empty-but-wired project skeleton and environment. No data processing and no model code.

**Pre-flight**
- Confirm that `D:` has more than 10 GB free.
- Set `PIP_CACHE_DIR=D:\pip_cache`, `TMP=D:\tmp`, and `TEMP=D:\tmp` for the install session. C: has only ~2 GB free, and the CUDA PyTorch wheel alone needs about that much.

**Steps**
1. Create the folder tree of Section 6.1 (empty module files with docstring headers only, `tests/fixtures/`, `docs/phase2/`, `docs/phase3/`). Do not touch `docs/phase1/` or `docs/planning/`.
2. Write `requirements.txt` with the pinned versions of Section 6.6 (PyTorch from the `cu126` index) plus `ipykernel` and `nbconvert`.
3. Create `D:\pattern_prj\.venv` with Python 3.12.10 and install the requirements (installation requires approval).
4. Write the config files (Revision 3.1: `project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`; prepared in the skeleton): split (train 118575, 118576, 118577, 118578, 128058; optional 117092; val 117093; test 128057, 132831), seeds, and placeholder parameter sections for each module.
5. Write `configs/data_sources.example.yaml`, and `configs/data_sources.yaml` with this machine's paths:
   - train/val MOT txt and clips on `F:\Football-Fatigue-Project\…`, including the special 117093 clip path
   - test paths under `D:\pattern_prj\data\raw\soccertrack-v2\mot\`
   - weights `F:\Football-Fatigue-Project\models\yolov8n.pt` and `yolov8s.pt`
   - calibration folders
6. Extend `.gitignore` with `configs/data_sources.yaml`, `*.npz`, `*.npy`, `pip_cache/`, `tmp/`.
7. Write a `README.md` setup section (environment, data catalog, test download, licences).

**Tests**
- `tests/test_environment.py` imports numpy, cv2, sklearn, torch (and reports `torch.cuda.is_available()`), ultralytics, trackeval, and pycocotools.
- `tests/test_config.py` loads both YAML files and checks that the split is disjoint and that all F: paths exist. This is a read-only existence check; skip it with a message on machines without F:.

**Acceptance**
- `pytest` passes.
- CUDA is visible on the GTX 1650 Ti.
- `git status` shows no files from `data/` or `outputs/` and no local `data_sources.yaml`.
- Nothing under `F:\` or `docs/phase1/` is modified.

---

## 10. Modularity for a four-person team (no assignment yet)

> Superseded by the final allocation in `docs/team/INTEGRATION_PLAN.md` (Revision 3.1). Kept for traceability.

The contracts in Section 4.2 let work packages proceed in parallel against synthetic fixtures. Integration happens only through the CSV/NPZ/JSON contracts and `config.yaml` sections. Each package has one owner-facing deliverable and a part of the notebook/report it can explain.

| Work package | Modules / tasks | Inputs → outputs | Relative effort | Notes |
|---|---|---|---|---|
| WP1 Data & EDA | `data_io`, `split`; T01, T03, T04 | MOT files, clips → Track tables, EDA figures, split table | Lighter | Independent; needs local data access |
| WP2 Segmentation & proposals | `segmentation`, `proposals`; T05, (T06 MWB-0) | frames → Proposal tables, stage figures | Heavier | Core of Approach A |
| WP3 Classification | `patches`, `features`, `classifiers`; T07–T09, (T20 in Phase 3) | Patch sets → models, E1 | Heavier | Can start on fixtures before WP2 finishes |
| WP4 Pretrained detector | `detector_yolo`; T11, (T17 in Phase 3) | frames → Detection tables | Moderate | Independent of Approach A |
| WP5 Evaluation & tracking | `metrics`, `evaluate`, `tracker`; T06, T13, (T19) | Detection/Track tables → Metrics JSON | Moderate–heavier | Works on GT tracks from day one (oracle) |
| WP6 Integration & reporting | `visualize`, notebook, report, slides; T14, T15, (T21 workload in Phase 3) | Metrics JSON, figures → deliverables | Lighter | Independent; consumes only contracts |

**Properties that support a fair split later:**
- Every package produces a demonstrable artefact and an explainable notebook/report section.
- Lighter packages (WP1, WP6) are independent of model training.
- Heavier packages (WP2, WP3) can be paired or split.
- WP4/WP5 can be grouped with either side.
- Conflicts are minimised because each package owns separate files.
- Team members on machines without the F: drive can work from small derived artefacts (Track/Detection CSVs, patch NPZ files) or download the needed clips themselves (≈ 3.2 GB for all train/val/test clips used).

---

## 11. Risks, contingencies and open blockers

### 11.1 Risks

| Risk | Contingency |
|---|---|
| HSV pitch mask or background subtraction fails | Fallbacks: keypoint polygon → manual polygon; running median; Lab chroma difference; perspective-scaled sliding windows (HOG+SVM). If proposal recall stays low, E1 becomes the main from-scratch result and E3 an explained weakness |
| Approach A performs poorly | Expected and explainable. E1 vs E3 separates classifier error from proposal error. MWB-0 guarantees an end-to-end number |
| Tracking takes longer than planned | Should-have only. The oracle E4 is small; drop detection-based E4 at the Monday cut line |
| GPU limits | Approach A runs on CPU; YOLOv8n with 640 tiles; Colab/Kaggle for Phase 3 fine-tuning |
| Training too slow | Larger frame stride, fewer patches, fewer epochs |
| Accidental test tuning | Frozen configuration hash + `test_log.md`; failure analysis on validation only |
| Paper benchmark not reproducible | Expected (no MOT numbers in the paper; GSR costs ~37 GPU-h per half). Reproduce dataset statistics; compare with the Challenge leaderboard with caveats |
| Calibration error (~2–3 m) | Workload reported with uncertainty; GT-track vs predicted-track comparison |

### 11.2 Open technical blockers

| # | Blocker | Owner | Resolution |
|---|---|---|---|
| B1 | HF gated access and token for the test download not yet confirmed | Team | Accept terms on the dataset page; create a read token; user runs T02 |
| B2 | **C: drive has ~2 GB free.** pip cache, temp unpacking, and the HF cache default to C: | Codex (T00) | Set `PIP_CACHE_DIR`, `TMP`, `TEMP`, and `HF_HOME` to D: before any install or download |
| B3 | F: may be a removable drive | Team | Confirm. If removable, approve copying the 2.3 GB of train/val files to D: |
| B4 | Test clip geometry unverified (132831 is likely 3840 px wide with a different aspect; 604 MB) | Codex (T01) | `check_data.py` reports resolution; size model h(y) and tiling are per-clip |
| B5 | Calibration keypoints may not share the clip's pixel space | Codex (T05/T21) | Overlay check; only affects the polygon fallback and Phase 3 |
| B6 | Other team members' machines lack the local data | Team | Work from derived artefacts or download ≈ 3.2 GB per machine |
| B7 | `ipykernel`/`nbconvert` versions not yet pinned (absent from the earlier environment) | Codex (T00) | Record versions at install |

---

## 12. Decisions and approvals

**Provisionally approved:**
- The technical direction (Approaches A and B, tracking as a should-have).
- Reuse of local data in place.
- Downloading only the test matches.
- The split.
- Reuse of the COCO YOLOv8n weights.
- Prior results as reference only.

**Still open (to confirm with the final approval):**
1. Copy `download_manifest.json` (a few KB) into `data/manifests/` for hash verification of the new downloads.
2. Download the optional second validation match 132877 (+0.31 GB)?
3. Include 117092 in training (optional; different geometry)?
4. HOG implementation: own NumPy implementation validated against `cv2.HOGDescriptor` (recommended: educational, no new dependency) vs `scikit-image`.
5. Phase 3 fine-tuning location: local GTX 1650 Ti vs Colab/Kaggle.
6. Include the third-party football detector as a clearly labelled extra row (default: no).
7. Approval for T00 to create the virtual environment and install the pinned packages.

---

## 13. Discussion defense notes

- **Why MOT clips?** They are the only human-annotated boxes in SoccerTrack v2 (GSR boxes are generated from pitch positions). They are the Challenge benchmark data, and seven of the ten were already verified locally.
- **Why segmentation?** The camera is static, so a per-pixel median of ~60 frames gives an empty-pitch background. Subtracting it inside an HSV pitch mask, then applying blur, a median filter, and opening/closing, yields player blobs. Connected components become proposals. There are no mask labels, so segmentation is measured by proposal recall.
- **Why a classifier?** Blobs include shadows, the ball, line artefacts, and unlabelled referees/staff. A 64×32 crop classifier makes a genuine binary decision, so confusion matrices, Sensitivity, and Specificity are meaningful.
- **HOG+SVM vs Small CNN:** hand-crafted gradient-orientation features (756-d) with a maximum-margin linear classifier, versus learned features from three convolution blocks. Same patches, same split.
- **Class imbalance:** background vastly outnumbers players. Handled with 1:3 sampling, class weights, and hard negatives, and evaluated with PR-AUC/F1. E2 shows the natural imbalance.
- **A classifier is not a detector:** it becomes one only with proposals, box generation, and NMS.
- **Tracking:** cost = 1 − IoU, optimal one-to-one assignment with the Hungarian algorithm, gating, and track birth/death. It is validated first on ground-truth boxes, so association quality is measured without detector errors.
- **Why not GS-HOTA 12.25?** It belongs to a different task: pitch-space, jersey/team/role-aware, over 45-minute halves. The paper points to the Challenge leaderboard (HOTA 0.60) for MOT.
- **HOTA:** √(DetA × AssA), averaged over IoU thresholds. IDF1 measures identity consistency; MOTA combines misses, false positives, and ID switches.
- **Data-format finding:** the released MOT files are 0-indexed with −1 metadata, unlike the official documentation. We verified this and normalise explicitly for TrackEval.
- **Workload boundaries:** distance, speed, acceleration, and heatmaps are external load indicators on 4-minute clips with ~2–3 m calibration error. There is no physiological fatigue or injury ground truth.

---

## 14. Sources
- Paper: https://arxiv.org/abs/2508.01802 (HTML v2)
- Dataset: https://huggingface.co/datasets/atomscott/soccertrack-v2 (public file listing via the HF API)
- Code: https://github.com/AtomScott/SoccerTrack-v2 (`README.md`, `baselines/`, `docs/format-mot.md`, `docs/calibration-findings.md`, `docs/TODO.md`, `docs/leaderboards/mot.json`, `src/evaluation/mot_hota.py`, `data_corrections/`)
- SoccerTrack Challenge 2025: https://sites.google.com/g.sp.m.is.nagoya-u.ac.jp/stc2025 · https://codalab.lisn.upsaclay.fr/competitions/22532
- GTATrack (Challenge winner): https://arxiv.org/abs/2602.00484
- Local evidence: `docs/phase1/`; `F:\Football-Fatigue-Project` (audited read-only; see `transfer_manifest.md`)
