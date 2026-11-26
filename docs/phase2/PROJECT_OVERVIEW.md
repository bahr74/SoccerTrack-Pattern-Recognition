# SoccerTrack — Comprehensive Project Overview

**Course:** Medical Pattern Recognition (SBES 426) — Faculty of Engineering, Biomedical Engineering Department, Cairo University  
**Dataset:** SoccerTrack v2 (`atomscott/soccertrack-v2`, release `v1.2`, Scott et al., 2025 — arXiv:2508.01802)  
**Author:** Saif Elshaaer (`Seifff1`)  
**Scope:** End-to-end technical overview of the SoccerTrack repository, architecture, Phase 1–Phase 2 implementation and validated results, and Phase 3 roadmap.

---

## 1. Project Objective & Phase Boundaries

The SoccerTrack project develops and compares vision-based pattern recognition pipelines for **football player detection and multi-object tracking (MOT)** in fixed, full-pitch panoramic video, as an accessible foundation for external-workload movement analysis.

The work is structured across three phases with strict scope boundaries:

| Phase | Scope & Deliverables | Status |
|---|---|---|
| **Phase 1** | Problem definition, literature review of SoccerTrack v2 (GSR & BAS), dataset schema analysis, class-imbalance review, and implementation roadmap. | Completed (`docs/phase1/`) |
| **Phase 2** | Data loading, integrity validation, leakage-safe match splitting, **Pipeline A (from-scratch)** and **Pipeline B (pretrained YOLOv8n + ByteTrack)**, shared data contracts, unified evaluation suite (**E1–E4**), visualizations, baseline notebook, report, and presentation. | Completed on validation split |
| **Phase 3** | Error/failure analysis, model & tracker improvements (YOLO fine-tuning, pitch filtering, appearance/Kalman tracking), explainability (Grad-CAM, SVM weights), full-clip tracking, frozen held-out test evaluation, camera homography pitch mapping, and **external workload indicators** (distance, speed, acceleration, heatmaps). | Planned (`explain.py`, `pitch.py`, `workload.py` stubbed) |

> **Scope note:** Phase 2 outputs are **image-space bounding boxes and temporary track IDs** (not player names or jersey numbers). Physical external-workload metrics are planned for Phase 3, and **no clinical fatigue or physiological injury diagnosis** is claimed because the dataset contains no physiological ground truth.

---

## 2. Dataset, Integrity Checks & Match-Level Split

### 2.1 SoccerTrack v2 Properties
* **Source:** Hugging Face `atomscott/soccertrack-v2` (gated, CC BY 4.0), pinned to revision `325bd4c57ee8e135ab652f3921d274526ada496b`.
* **Video format:** Fixed panoramic stitched 4K footage at **`4096 × 1080` pixels**, **`25 fps`** (`40 ms` per frame).
* **MOT annotations:** 10-column headerless CSV files (`frame,id,x,y,w,h,c7,c8,c9,c10`) using **0-based frame indices**, top-left full-frame pixel coordinates `(x, y, w, h)`, and unused trailing fields (`-1`). Every inspected annotated frame contains **`22` player boxes**.
* **Core visual challenge:** Because the full pitch fits in a single wide panorama, players occupy very small pixel regions (published median scale ≈ `25 × 55 px`), making both detection and tight localization challenging.

### 2.2 Leakage-Safe Match Split
Randomly splitting frames from the same match leaks near-identical consecutive frames (`40 ms` apart) across train and validation. Therefore, all splits in `configs/project.yaml` are strictly separated **at the match level**:

| Split | Match IDs | Annotated Frames | Usage in Phase 2 |
|---|---|---|---|
| **Training** | `118575`, `118576`, `118577`, `118578`, `128058` | `29,956` | Patch extraction, background/size models, local HOG/SVM & CNN training |
| **Validation** | `117093` | `5,976` | Model/hyperparameter selection and E1–E4 reporting |
| **Held-out Test** | `128057`, `132831` | Protected | Held out; **never opened or evaluated** in Phase 2 |
| **Optional (Disabled)** | `117092` (train), `132877` (val) | Disabled | Excluded (`117092` has different camera geometry) |

* **Approved 6-match subset:** `35,932` annotated frames and `790,504` ground-truth boxes (verified from annotations and video headers).
* **Data cleaning (`src/soccertrack/data_io.py` & `split.py`):** Rejects malformed, missing, and non-finite fields; validates integer IDs; removes duplicate rows; clips boxes to frame bounds; drops degenerate boxes (`w <= 1` or `h <= 1`); rejects conflicting `(frame, id)` records; and enforces leakage/test-access guards (`require_test_access`, `log_test_access`). Zero geometry integrity issues were found in the approved subset.

---

## 3. Repository Architecture & Shared Contracts

All modules are decoupled and communicate **exclusively** through validated file schemas defined in `docs/team/SHARED_CONTRACTS.md` and enforced by `src/soccertrack/contracts.py`.

### 3.1 System Data Flow

```text
                        ┌──────────── Pipeline A: From Scratch (Member 1) ─────────────┐
  Video + MOT ──► data  │ Median BG + HSV/Lab seg ──► Proposals ──► Rules / HOG+SVM /  │──► Detections ──► IoU + Hungarian ──► Tracks ─┐
  (0-based)       layer │                                            Small CNN + NMS   │                                               │
                        └──────────────────────────────────────────────────────────────┘                                               ├──► Evaluation E1–E4
                        ┌──────────── Pipeline B: Pretrained (Member 2) ───────────────┐                                               │    (Member 3)
                  ─────►│ Native 1280×1080 tiles (128 px overlap) ──► YOLOv8n ──► NMS  │──► Detections ──► ByteTrack ────────► Tracks ─┘         │
                        └──────────────────────────────────────────────────────────────┘                                                         ▼
                                                                                                                               Tables, Figures, Notebook & Report
                                                                                                                               (Members 3 & 4; Phase 3 Workload)
```

### 3.2 Directory Layout & Module Ownership

| Directory / File | Purpose | Primary Owner |
|---|---|---|
| `src/soccertrack/data_io.py`, `split.py` | MOT/video I/O, sequential 0-based decoding, match splits, leakage guards | Member 1 |
| `src/soccertrack/segmentation.py`, `proposals.py`, `patches.py`, `features.py`, `classifiers.py`, `detector_scratch.py`, `tracker_iou.py` | Complete from-scratch detection and IoU/Hungarian tracking pipeline | Member 1 |
| `src/soccertrack/detector_yolo.py`, `tracker_bytetrack.py` | Tiled COCO YOLOv8n inference, coordinate merging, global NMS, ByteTrack adapter | Member 2 |
| `src/soccertrack/contracts.py`, `metrics.py`, `evaluate.py`, `compare.py`, `explain.py` | Schema validation, IoU/classification/detection/TrackEval metrics (E1–E4), comparison tables, Phase 3 explainability | Member 3 |
| `src/soccertrack/config.py`, `visualize.py`, `pitch.py`, `workload.py` | YAML config loading, deterministic `config_sha256`, seeds, EDA/overlay/stage plots, Phase 3 pitch & workload | Member 4 |
| `configs/` | `project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`, `data_sources.example.yaml` | Shared (one owner per file) |
| `scripts/` | CLI runners: `check_data.py`, `download_test_matches.py`, `run_scratch.py`, `run_yolo.py`, `run_evaluation.py`, `run_visualization.py` | Per-member |
| `tests/` | Synthetic `fixtures/` with hand-verified expectations + unit/contract tests (`pytest`) | Per-member |
| `notebooks/` & `docs/` | `phase2_baseline.ipynb`, `phase3_final.ipynb`, planning docs, team workflow guides, Phase 1–3 reports | Member 4 / Team |

### 3.3 Core Integration Contracts
* **Standardized tables (UTF-8 CSV + `.meta.json` sidecar):**
  * **Detection table:** `frame, x, y, w, h, score` (full-frame pixels, post-NMS, retaining all boxes with `score >= 0.05` so PR curves, AP, and ByteTrack low-score association work properly).
  * **Track table:** `frame, track_id, x, y, w, h, score` (shared by ground truth and both trackers).
  * **Proposal table:** `frame, x, y, w, h, area, fill_ratio, score` (pre-NMS, Approach A).
  * **Patch set (`.npz`) & Patch scores:** `64 × 32 × 3` RGB `uint8` crops with source metadata and labels (`gt`, `random`, `partial`, `proposal`).
* **Standardized framesets (`configs/evaluation.yaml`):**
  * `eval10` (`stride=10`, 598 frames on val `117093`): E2 proposal & E3 detection evaluation.
  * `win60s` (`frames 0..1499`, 60 s at 25 fps): Phase 2 E4 tracking evaluation.
  * `full` (all frames): Phase 3 full-clip tracking and workload.
  * `smoke` (100 frames) & `fixture3` (3 frames): integration and unit testing only.

---

## 4. Pipeline A — From-Scratch Detection & Tracking

### 4.1 Candidate Proposal Generation
1. **Median Background Estimation:** Computes a clip-level median background image across sampled frames.
2. **Segmentation & Morphology:** Runs at half scale (`work_scale = 0.5`) using Lab color-space foreground differencing combined with an HSV green-pitch mask and morphological opening/closing.
3. **Perspective-Aware Filtering & Splitting:** Fits an expected player height vs. image row model `h_expected(y)` on **training GT only**, filters connected components outside the height tolerance, splits wide merged components, and rescales boxes to full `4096 × 1080` coordinates.
4. **Proposal Optimization (Validation-Selected):** Lowering `diff_threshold` (`25 → 18`), morphological kernels (`open 3 → 1`, `close 7 → 5`), `min_area_px` (`60 → 40`), and broadening `size_tolerance` (`[0.5, 2.0] → [0.35, 2.0]`) substantially improved candidate recall without retraining the downstream classifiers.

### 4.2 Local Patch Dataset & Imbalance Handling
* Ground-truth player boxes are expanded by **10% context per side** and resized with `cv2.INTER_AREA` to **`64 × 32` RGB**.
* Negatives are sampled at a **3:1 negative-to-positive ratio** from random pitch background (`IoU < 0.1`), partial player crops (`0.1 <= IoU < 0.4`), and unmatched candidate proposals (hard negatives).
* **Sample sizes:**
  * **Train (`39` sampled frames):** `858` player + `2,574` non-player = **`3,432` patches** (`2,782` unique pixel rows; `650` repeated negative rows retained for historical reproducibility).
  * **Validation (`8` sampled frames):** `176` player + `528` non-player = **`704` patches** (`580` unique pixel rows; `124` repeated negative rows).

### 4.3 Baseline Verifiers / Classifiers
1. **Rule-Based Baseline (`scratch_mwb0`):** Non-learned classical reference combining aspect-ratio shape score (`h/w ∈ [1.2, 4.0]`), foreground fill ratio (normalized by `0.3`), and perspective size agreement: `score = clip(shape × fill × agreement, 0, 1)`, followed by thresholding and NMS (`IoU = 0.50`).
2. **HOG + Linear SVM (`scratch_hog_svm`):**
   * Custom grayscale central-difference HOG with `8 × 8` px cells (`8 × 4` grid), overlapping `2 × 2`-cell blocks (`21` blocks), `9` unsigned orientation bins, and L2-Hys normalization → **`756` features** per patch (`3,432 × 756` training matrix).
   * Balanced `LinearSVC` (squared hinge loss + L2; class weights `2.0` positive / `0.667` negative) with 3-fold train-only stratified sigmoid probability calibration.
   * Grid search over `C ∈ {0.01, 0.1, 1.0, 10.0}` selected **`C = 0.01`** by validation AP.
3. **Compact CNN (`scratch_cnn`):**
   * Input: normalized `3 × 64 × 32` RGB tensor (train-only mean/std; train augmentation: horizontal flip `p = 0.5`, brightness/contrast `±0.2`; no validation augmentation).
   * Architecture: 3 convolutional blocks (`Conv3×3` stride 1 pad 1 + BatchNorm + ReLU + `MaxPool2×2`) with `16`, `32`, and `64` channels (`480`, `4,704`, and `18,624` params), followed by Global Average Pooling, `Dropout(0.3)`, and a 1-logit Linear head (`65` params) → **`23,873` trainable parameters**.
   * Training: weighted `BCEWithLogitsLoss` (`pos_weight = 3.0`), Adam (`lr = 0.001`, `weight_decay = 0.0001`), batch size `128`, `8` epochs. **Epoch 7** was selected by peak validation AP (`0.9788` vs. `0.9634` at epoch 8).

### 4.4 Pipeline A Tracker (`iou_hungarian`)
* Pure from-scratch association using the Hungarian algorithm on cost `1 − IoU`, minimum IoU gate **`0.30`**, track confirmation after **`n_init = 3`** hits, and termination after **`max_age = 15`** missed frames (no Kalman motion model in Phase 2).

---

## 5. Pipeline B — Pretrained Tiled YOLOv8n & ByteTrack

### 5.1 Overlapping Tiled YOLOv8n (`yolov8n_coco`)
* **Model:** COCO-pretrained **YOLOv8n** (`3,157,200` parameters; Conv/C2f/SPPF backbone, feature-fusion neck, multi-scale heads at strides `8`, `16`, `32`), filtered to **class `0` (person)** only. **No SoccerTrack fine-tuning** was performed in Phase 2.
* **Why tiling:** Resizing a `4096 × 1080` panorama directly shrinks tiny players below reliable feature resolution. The frame is sliced into native-resolution **`1280 × 1080` overlapping tiles** (`128 px` nominal overlap; edge-aligned final tile overlaps more), run at `imgsz = 1280`, translated back to full-frame pixel coordinates, and deduplicated via global NMS (`IoU = 0.50`, source floor `0.05`, reporting confidence `0.50`, pitch filtering off).
* **Development selection:** This overlapping configuration achieved `F1 = 0.8352` on `33` disjoint development frames prior to E3 reporting.

### 5.2 Pipeline B Tracker (`bytetrack`)
* Adapter around Ultralytics `BYTETracker`, adding **Kalman-filter motion prediction**, high-confidence first-stage matching, and **low-confidence second-stage recovery** for temporarily occluded or low-score players.
* **Selected real-pipeline settings** (chosen by maximum validation HOTA): `track_high_thresh = 0.40`, `new_track_thresh = 0.40`, `track_low_thresh = 0.10`, `track_buffer = 30`, `match_thresh = 0.80`, `fuse_score = true`. Controlled oracle/noisy experiments use the starting thresholds (`0.25`).

---

## 6. Phase 2 Evaluation Suite & Validated Results

All metrics below come from the accepted Phase 2 validation run (`phase2_20261125_01`) on validation match **`117093`**. Held-out test matches (`128057`, `132831`) remain untouched.

### E1 — Patch Classification (`704` validation patches, threshold `0.50`)

| Method | Accuracy | Precision | Sensitivity | Specificity | F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Majority baseline** | 0.7500 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.2500 |
| **HOG + SVM** | **0.9418** | **0.8902** | 0.8750 | **0.9640** | **0.8825** | 0.9779 | 0.9155 |
| **Small CNN** | 0.8764 | 0.6718 | **0.9886** | 0.8390 | 0.8000 | **0.9923** | **0.9788** |

* **Takeaway:** HOG/SVM leads fixed-threshold F1 at `0.50` (`TP=154, FP=19, FN=22, TN=509`), while CNN achieves superior probability ranking (`ROC-AUC 0.9923`, `AP 0.9788`) and misses only `2` players (`TP=174, FP=85, FN=2, TN=443`).

### E2 — Proposal Generation (`598` `eval10` frames, `13,156` GT boxes)

| Proposal Variant | Total Proposals | Recall @ IoU 0.30 | Recall @ IoU 0.50 |
|---|---:|---:|---:|
| **Starting** | 14,615 | 0.8070 | 0.5412 |
| **Selected** | 17,360 | **0.8867** | **0.7083** |

* **Takeaway:** Proposal geometry tuning raised GT coverage at IoU `0.50` by **+16.71 percentage points**. On selected natural proposals, HOG/SVM and CNN achieved classification F1 of `0.8634` and `0.8527`, respectively.

### E3 — Full-Frame Detection (`598` `eval10` frames, `13,156` GT boxes, score `>= 0.50`, IoU `0.50`)

| Detector | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Rules (`scratch_mwb0`)** | 5,881 | 1,904 | 7,275 | 0.7554 | 0.4470 | 0.5617 | 0.5104 | 0.2344 |
| **HOG + SVM (`scratch_hog_svm`)** | 7,977 | **1,360** | 5,179 | **0.8543** | 0.6063 | 0.7093 | 0.6248 | 0.3020 |
| **Small CNN (`scratch_cnn`)** | 9,070 | 3,061 | 4,086 | 0.7477 | 0.6894 | 0.7174 | 0.6327 | 0.2888 |
| **YOLOv8n Tiled (`yolov8n_coco`)** | **11,127** | 2,983 | **2,029** | 0.7886 | **0.8458** | **0.8162** | **0.7739** | **0.5552** |

* **Takeaway:** **Small CNN** is the strongest scratch detector by F1 (`0.7174`), while **HOG/SVM** has the highest precision (`0.8543`). Overall, **tiled YOLOv8n** leads decisively in recall (`0.8458`), F1 (`0.8162`), AP50 (`0.7739`), and localization quality (`AP50:95 = 0.5552`).

### E4 — Multi-Object Tracking (`1,500` consecutive `win60s` frames = 60 s, `33,000` GT observations)

Controlled tracking tests isolate association quality from detector errors using exact GT boxes (`gt_oracle`) and perturbed GT boxes (`gt_noisy`: `10%` random drop + Gaussian jitter `σ = 2 px`, seed `42`):

| Input + Tracker | HOTA | IDF1 | MOTA | ID Switches |
|---|---:|---:|---:|---:|
| **Oracle + IoU/Hungarian** | 0.9680 | 0.9648 | 0.9984 | 4 |
| **Oracle + ByteTrack** | 0.9699 | 0.9772 | 0.9998 | 7 |
| **Noisy GT + IoU/Hungarian** | 0.2525 | 0.2254 | 0.7732 | 1,722 |
| **Noisy GT + ByteTrack** | **0.4744** | **0.5986** | **0.8038** | **669** |
| **CNN + IoU/Hungarian** | 0.2551 | 0.2680 | **0.4058** | 465 |
| **YOLO + Selected ByteTrack** | **0.4321** | **0.4327** | 0.3255 | **172** |

* **Takeaway:** Under noisy detections, ByteTrack nearly doubles HOTA (`0.4744` vs. `0.2525`) and cuts ID switches by more than 60%. On real detections, **YOLO + Selected ByteTrack** slashes missed detections (`12,071 → 3,711`) and ID switches (`465 → 172`), leading HOTA (`0.4321`) and IDF1 (`0.4327`). Its lower MOTA (`0.3255` vs. `0.4058`) reflects higher false positives (`18,374` vs. `7,071`), pinpointing false-positive reduction as a top Phase 3 priority.

### Reference Paper Comparison (Scott et al., 2025)
* The SoccerTrack v2 paper evaluates **Game-State Reconstruction (GSR)** across 10 matches / 20 full halves using RF-DETR + ViTPose + PRTReID + BoT-SORT + ViTSTR + camera homography, reporting **GS-HOTA** of `12.25` (all halves) and `10.96` (held-out test halves) on a `0–100` scale.
* Because GS-HOTA measures metric pitch-space localization and player attributes (team/role/jersey) over full halves while Phase 2 measures image-space boxes and temporary track IDs on one 60-s validation clip, the scores are **not directly comparable** and no numerical superiority is claimed.

---

## 7. Visualization, Testing & Reproducibility

* **Visualization (`src/soccertrack/visualize.py`, `scripts/run_visualization.py`):** Generates EDA figures (`boxes_per_match`, `box_size_hist`, `height_vs_row`, `track_length`, `data_issues`), 2×2 scratch segmentation stage grids, bounding-box overlays, vertical side-by-side pipeline comparisons (preserving `4096 × 1080` legibility), and scaled comparison MP4 videos.
* **Reproducibility & Engineering Hygiene:**
  * Deterministic `config_sha256()` hashes the four committed YAML files (`project.yaml`, `scratch.yaml`, `yolo.yaml`, `evaluation.yaml`) across OS line endings.
  * Read-only root guards prevent accidental writes into raw dataset directories; `configs/data_sources.yaml`, datasets, model weights, and generated `outputs/` are git-ignored.
  * Unit and contract tests run completely offline via `pytest` against synthetic fixtures in `tests/fixtures/`.

---

## 8. Key Limitations & Phase 3 Roadmap

### Current Phase 2 Limitations
1. Single validation match (`117093`); independent held-out test performance (`128057`, `132831`) is not yet evaluated.
2. Local patch classifiers were trained on only `39` sampled frames and retained repeated negative pixel rows for historical reproducibility.
3. Proposal and tracker hyperparameters were selected on validation (`117093`), not an independent test split.
4. Custom HOG numerical parity against OpenCV's reference HOG API remains unresolved due to environment API availability.
5. Median background estimation uses frames across the clip (non-causal offline estimation, not real-time streaming).
6. High false-positive rate in `yolov8n_coco + bytetrack` (reducing MOTA), small/occluded player misses, and identity fragmentation over long sequences.

### Phase 3 Planned Improvements
1. **Failure Analysis & Explainability (`explain.py`):** Quantify error modes on train/val (tiny players, off-pitch false positives, occlusions) and apply Grad-CAM and SVM weight inspection.
2. **Detector Improvements:** Investigate football-specific/fine-tuned YOLO (`yolov8n_ft` / `yolov8s_ft`), enable/refine pitch-mask filtering to eliminate off-pitch false positives, and improve small-player localization.
3. **Tracker Improvements:** Explore Kalman filtering for `iou_hungarian_v2`, tuned ByteTrack thresholds, and appearance-based re-identification to reduce identity fragmentation across full 4-minute clips.
4. **Configuration Freeze & Final Test Run:** Freeze all parameters and run the single logged evaluation on held-out test matches `128057` and `132831`.
5. **Pitch Mapping & External Workload (`pitch.py`, `workload.py`):** Calibrate camera homography into `105 m × 68 m` metric pitch coordinates, smooth trajectories (Savitzky-Golay), and compute external workload indicators (total distance, speed zones, high-speed/sprint counts, acceleration events, and spatial heatmaps) with explicit uncertainty bounds.
