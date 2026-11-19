# Member 1 — Data Processing and the Complete From-Scratch Pipeline (Pipeline A)

Branch: `feature/scratch-pipeline` · AI prompt: [`prompts/MEMBER_1_AI_PROMPT.md`](prompts/MEMBER_1_AI_PROMPT.md)
Read first: [`../planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1), [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md), [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md), [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

---

## 0. START HERE — Your first working session

Your first task: **M1: real MOT reading, cleaning, validation, sequential video decoding and leakage-safe splitting** (`data_io.py` core + `split.py`). Read §0.3 below before starting.

Data mode for this task: Mode 1 for the logic; real-file checks in Mode 3 (lead machine, the MOT files already exist on F:) or Mode 2 (`mot/117093.txt` + `mot/clips/117093.mp4`, ≈ 0.31 GB) (modes: [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §2; your data plan: §3).

1. **Clone and open** (only after the approved skeleton is merged into `main`):
   ```powershell
   git clone https://github.com/bahr74/SoccerTrack-Pattern-Recognition.git pattern_prj
   cd pattern_prj
   code .
   ```
2. **Read, in this order:**
   1. [`README.md`](../../README.md);
   2. [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §1–3 (dataset orientation and your data plan);
   3. this guide §0, §2–5 and §9;
   4. [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md): §1 (conventions), §3 (Track table), §4 (Detection table), §5–7, §11 (EDA stats), §14 (test access), §15 (fixtures);
   5. [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0, §2–4;
   6. [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §1–5.
3. **Create your branch:**
   ```powershell
   git switch main; git pull origin main
   git switch -c feature/scratch-pipeline
   ```
4. **Know your files:**
   - **Owned** (edit only these): `src/soccertrack/{data_io,split,segmentation,proposals,patches,features,classifiers,detector_scratch,tracker_iou}.py`, `scripts/run_scratch.py`, `configs/scratch.yaml`, your seven test files, your two `sections/` files.
   - **Read-only** (import or read, never edit): `config.py`, `contracts.py`, `metrics.py`, `evaluate.py`, other configs, `tests/conftest.py`, `tests/fixtures/**`, notebooks, `docs/team/*`.
   - **Off-limits:** `detector_yolo.py`, `tracker_bytetrack.py` (never even import them: Pipeline A stays independent); `docs/phase1/**`; everything under `F:\Football-Fatigue-Project` (read-only).
5. **Python environment** (only after the team has approved installation): [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §4.
6. **Initial fixture checks** (✅ work immediately, no data needed):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pytest                                  # a few contract/fixture checks pass; most tests are SKIPPED placeholders
   pytest -rs tests/test_data_io.py tests/test_split.py                 # shows the skipped acceptance tests of your first milestone
   python scripts/run_scratch.py --help             # argument parsing only; the commands themselves are not implemented yet
   ```
   A green run with skips proves nothing about your code. Any call into an unimplemented function raises `NotImplementedError("TODO(Member n, Mk)")`.
7. **Understand your first milestone:** §9 (milestones) and §0.2–0.3 below.
8. **Implement only that milestone.** Do not start the next one in the same branch state.
9. **Replace the skipped tests of the milestone** with real tests (remove the `@pytest.mark.skip`, write the assertions):
   - `test_data_io.py`: `read_mot_raw` on `mot_raw_sample.txt`; `mot_to_tracks` equals `tracks_gt.csv`; dirty file gives 11 rows and issues {1, 1, 1, 1}; `iter_frames` on a tiny synthetic video
   - `test_split.py`: `matches_for`, `assert_disjoint`, `require_test_access`, `log_test_access`
   - the data-marked test (decoded frames = max frame + 1) is run on a machine with data (Mode 2/3) with `pytest -m data`
10. **Run and inspect:**
    ```powershell
    pytest tests/test_data_io.py tests/test_split.py -v
    pytest -rs                              # no skip with your milestone ID may remain
    ```
11. **Document:** docstrings in your modules, and a short note of what was done in `docs/phase2/sections/member_1_scratch.md`.
12. **Report blockers** to the owner of the missing piece, through the team chat or the PR, never by editing their file ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §9).
13. **Commit and open a PR** (only when your team has authorised it):
    ```powershell
    git status
    git add <your owned files only>
    git diff --cached --name-only           # nothing outside your owned files; no data, weights, outputs or credentials
    git commit -m "M<k>: <what was done>"
    git push -u origin feature/scratch-pipeline
    ```
    Then open a pull request into `main` with the checklist in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §5.

**Commands that work only later:**

| Command | Works |
|---|---|
| `python scripts/run_scratch.py eda` | ⏳ after M2 |
| `python scripts/run_scratch.py segment …`, `patches …`, `train-svm …` / `train-cnn …` / `score-patches …`, `detect …`, `track …` | ⏳ after M3, M4, M5/M6, M7, M8 |
| `pytest -m data tests/test_data_io.py` | ⏳ after M1, on a machine with data (and after Member 4's `config.load_data_sources`) |

### 0.1 Your data
Data needed now → source → local configuration → verification → data needed later: see your row in [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §3. Never put datasets, weights, outputs or credentials into Git.

### 0.2 Dependency and independence matrix

| Milestone | Required input | Exists now? | Owner / provider | If not ready yet, use | Proof of completion | Hand-off |
|---|---|---|---|---|---|---|
| **M1** data_io + split | Raw MOT fixtures; real MOT txt + val clip; `configs/project.yaml` split | Yes (fixtures; F: on the lead machine; project.yaml committed) | Member 3 (fixtures); dataset; Member 4 (config) | `config.load_config` (Member 4, M0) may still raise: in tests, build the `cfg` dict with `yaml.safe_load(open('configs/project.yaml'))`. `contracts.validate_tracks` (Member 3, M0) may still raise: compare with `tracks_gt.csv` now and add the call when M0 is merged (same target time) | M1 skips replaced and passing; `pytest -m data` on a data machine | Merged `data_io` + `split` → Members 2, 3, 4 |
| **M2** EDA | MOT files of the 7 local matches | Lead machine only (or your Mode 2 subset) | Dataset | Develop `compute_eda_stats` on fixtures; run the full statistics on the lead machine | `stats.json` validates against Contracts §11 | `outputs/eda/stats.json` → Member 4 (figures) |
| **M3** segmentation, proposals, MWB-0 | Val frames; `metrics.iou_matrix` | Frames: lead machine/Mode 2. IoU: after Member 3 M0 | Member 1 (M1); Member 3 | `synthetic_frame` fixture for unit tests | Unit tests; proposal recall@0.5/0.3 on val | Proposal tables, `scratch_mwb0` Detection tables → Member 3 (E2/E3), Member 4 (stage figures) |
| **M4–M6** patches, HOG+SVM, CNN | Train/val frames + GT | Lead machine (Mode 3) or Mode 2 train clips + labels (≈ 1.47 GB) | Dataset; your M1 | `synthetic_patches` fixture | `validate_patch_set`; E1 on val run by Member 3's evaluator | Patch sets, patch score tables → Member 3 (E1) |
| **M7** scratch detector | Classifiers, proposals | Your own M3–M6 | Member 1 | — | E2/E3 on val via Member 3 | `scratch_hog_svm`, `scratch_cnn` Detection tables → Members 3, 4 |
| **M8** IoU + Hungarian tracker | Detection tables (`gt_oracle`/`gt_noisy`, your detector) | Oracle tables after Member 3 M2 | Member 3 | `tests/fixtures/detections_sample.csv` and `tracks_gt.csv` (drop `track_id`) | Tracker unit tests; E4 on val via Member 3 | `<detector>+iou_hungarian` Track tables → Members 3, 4 |

Parallel work is allowed, but integration checkpoints cannot be skipped. Each milestone that others depend on is merged into `main` before they rely on it ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0 and §4).

### 0.3 Primer for M1: MOT files, frames and splits

- **What an MOT annotation file is.**
  - One plain-text CSV per match (`mot/<match>.txt`, ≈ 6 MB).
  - One line per player per frame: about 22 × 6,000 ≈ 132,000 lines.
  - These are the **human-annotated** labels; read [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §1.2–1.3 first.
- **The ten columns.** `frame, id, x, y, w, h, c7, c8, c9, c10`:
  - `frame`: 0-based frame index;
  - `id`: player 1–22, constant over the clip;
  - `x, y`: top-left corner in full-frame pixels;
  - `w, h`: box size in pixels;
  - `c7`–`c10`: unused, −1 everywhere in the real files we checked.
  - `read_mot_raw` keeps all ten columns and only checks that exactly ten exist.
- **Frame ↔ video.**
  - `frame = k` is the k-th image produced by **sequential** decoding from the start of the clip.
  - Check: the decoded frame count must equal `max(frame) + 1`. For 117093, the last line is `5975,22,…` and there are 5,976 decoded frames.
  - Never seek by timestamp (the 117093 container starts at 0.979 s).
- **Raw → Track table** (`mot_to_tracks`; contract in [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md) §3):
  1. drop exact duplicate rows;
  2. drop degenerate boxes (`w ≤ 1` or `h ≤ 1`);
  3. clip boxes to the image size;
  4. rename `id` → `track_id`, set `score = 1.0`, keep `frame,track_id,x,y,w,h,score`, sort by `frame, track_id`.
- **Issues to count and report** (the returned `issues` dict):
  - `duplicate_rows`, `degenerate_boxes`, `clipped_boxes`;
  - `frames_not_expected`: frames whose box count ≠ 22 after cleaning.
  - Also check that there are no frame gaps and no repeated `(frame, track_id)`.
  - Fixture check: `tests/fixtures/mot_raw_dirty.txt` (3 boxes expected per frame, 640 × 360 image) → 11 rows, issues {1, 1, 1, 1}.
  - Real-data observation for M2: `118578.txt` is 4.5 MB while the others are ≈ 6.2 MB. EDA must find out why (fewer annotated frames or fewer boxes) and report it.
- **Why split by match.**
  - Consecutive frames are almost identical: same players, kits, camera and lighting.
  - A frame-level split would put near-copies of test images into training, and the result would measure memorisation.
  - So whole matches are assigned to train, validation or test (`configs/project.yaml`, frozen). `split.require_test_access` blocks the test split unless `--frozen-test` is given.
- **Running the first reader** (⏳ only after you implement it):
  ```powershell
  python -c "from soccertrack.data_io import read_mot_raw; print(read_mot_raw('tests/fixtures/mot_raw_sample.txt').head())"
  pytest tests/test_data_io.py tests/test_split.py -v
  pytest -m data tests/test_data_io.py          # data machine only, after Member 4's config M0 is merged
  ```
- **Order after M1:** EDA (M2) → segmentation + proposals + MWB-0 (M3) → patches (M4) → HOG + SVM (M5) → Small CNN (M6) → end-to-end scratch detector (M7) → independent IoU + Hungarian tracker (M8).
- **On the lead machine,** the real train/val files are on `F:\Football-Fatigue-Project`. Read them in place and never write, move or rename anything there.

---

## 1. Project overview and scientific objective
- **Task:** detect and track the 22 players in fixed panoramic football video from **SoccerTrack v2** (release v1.2, MOT subset: one 4-minute clip per match, 25 fps, about 4096 × 1080 px). The long-term goal is to compute **external workload indicators** (distance, speed, acceleration, heatmaps). These are not physiological fatigue measures.
- **Two complete pipelines** are compared under identical evaluation:
  - **Pipeline A (yours):** classical, explainable, trained from scratch. Segmentation → region proposals → HOG+SVM / Small CNN → player detection → IoU + Hungarian tracker.
  - **Pipeline B (Member 2):** tiled COCO-pretrained YOLOv8n → ByteTrack.
- **Scientific question for you:** how far can course techniques (segmentation, morphology, hand-crafted features, classical and small CNN classifiers, optimal assignment) go on tiny (~20 × 45 px) players, and where exactly do they fail (proposal stage vs. classification stage vs. association)?

## 2. Your scope

**Phase 2 (must-have unless marked):**
1. Dataset loading, parsing, cleaning and EDA statistics.
2. Train/validation/test split and leakage guards; test-access log.
3. Pitch segmentation, background subtraction, morphology, region proposals; proposal recall on validation.
4. MWB-0: rule-based verifier detector (the first end-to-end number, Fri 20 Nov).
5. Patch extraction (1:3 imbalance handling) and HOG feature extraction (implemented in NumPy).
6. HOG + Linear SVM (MWB-1) and the Small CNN classifier.
7. End-to-end scratch detector (proposals → classifier → NMS) with Detection and scored Proposal tables.
8. *(Should-have)* From-scratch IoU + Hungarian tracker, validated on `gt_oracle` → `gt_noisy` → your detections.
9. Test-split artefacts in the frozen run; your report section.

**Phase 3:** improve the scratch detector (hard-negative mining, merged-blob splitting or perspective-scaled sliding windows, CNN tuning) as `scratch_*_v2`, and the tracker (Kalman constant-velocity prediction, two-stage association, small-box-aware gating, short-gap stitching) as `iou_hungarian_v2`, all justified by Member 3's validation failure analysis.

## 3. Algorithms, methods and concepts
- **MOT parsing:**
  - 10 columns; frames 0-based; columns 7–10 are −1 (verified on the real files).
  - Cleaning: duplicates, degenerate boxes, clipping, per-frame counts.
- **Sequential decoding:** `cv2.VideoCapture` with `grab()`/`retrieve()`; frame k = k-th decoded frame. Never seek by timestamp (117093 has a 0.979 s container offset).
- **Pitch mask:** HSV thresholding on the median background → largest connected component → hole filling and closing. Fallbacks: polygon from calibration keypoints (check pixel space), or a manual polygon.
- **Background model:** per-pixel median of ~60 frames sampled across the clip (static camera). Fallback: running median.
- **Foreground:** Gaussian blur → absolute difference (Lab or gray) → threshold → median filter → opening → closing → small vertical dilation.
- **Proposals:**
  - Connected components (`cv2.connectedComponentsWithStats`), rescaled from `work_scale` to full resolution.
  - Size filter with the perspective model h(y) = a·y_bottom + b, fitted by least squares on **train** GT only.
  - Split blobs wider than 1.6 × the expected width.
- **Patches:** 64 × 32 RGB crops with 10 % context. Positives = GT; negatives = random (IoU < 0.1), partial (0.1–0.4), unmatched proposals; ratio 1:3.
- **HOG:** gradients [−1, 0, 1]; 8 × 8 cells; 9 unsigned orientation bins (0–180°) with bilinear vote interpolation; 2 × 2-cell blocks with stride 1 cell; L2-Hys normalisation (clip 0.2). Gives 7 × 3 × 36 = **756** features. Check against `cv2.HOGDescriptor` within a documented tolerance.
- **Linear SVM:** `sklearn.svm.LinearSVC(class_weight="balanced")`; choose `C` on validation PR-AUC; Platt calibration (`CalibratedClassifierCV`, fitted on train folds) for scores in [0, 1].
- **Small CNN (PyTorch):**
  - 3 × (Conv3×3 → BatchNorm → ReLU → MaxPool2) with 16/32/64 channels → global average pooling → dropout → linear(1). About 25k parameters (< 100k required).
  - Weighted BCE; Adam; light augmentation (horizontal flip, brightness/contrast); early stopping on validation PR-AUC.
  - Mean/std computed on **train** patches only.
- **NMS:** `torchvision.ops.nms`.
- **IoU + Hungarian tracker:**
  - cost = 1 − IoU; `scipy.optimize.linear_sum_assignment`; reject IoU < `iou_gate`.
  - Track lifecycle: tentative → confirmed after `n_init` hits → deleted after `max_age` misses. Output confirmed tracks only.

## 4. Files you own (edit only these)
- `src/soccertrack/`:
  - foundation: `data_io.py`, `split.py`
  - Approach A: `segmentation.py`, `proposals.py`, `patches.py`, `features.py`, `classifiers.py`, `detector_scratch.py`
  - Pipeline A tracker: `tracker_iou.py`
- `scripts/run_scratch.py`
- `configs/scratch.yaml`
- `tests/`: `test_data_io.py`, `test_split.py`, `test_segmentation_proposals.py`, `test_patches_features.py`, `test_classifiers.py`, `test_detector_scratch.py`, `test_tracker_iou.py`
- `docs/phase2/sections/member_1_scratch.md`, `docs/phase3/sections/member_1_scratch.md`

## 5. Shared files you may read/import but must not edit
- `config.py` (Member 4)
- `contracts.py`, `metrics.py` (`iou_matrix`), `evaluate.py` (Member 3)
- all `configs/*.yaml` except `scratch.yaml`
- `tests/conftest.py`, `tests/fixtures/**`
- `docs/team/*`, `docs/planning/*`, notebooks
- **Never** read or import `tracker_bytetrack.py` or `detector_yolo.py`: Pipeline A must stay independent.

## 6. Inputs and outputs

| You read | You write (contract section) |
|---|---|
| Raw MOT txt + clips (via `config.match_paths`) | `outputs/eda/stats.json`, `outputs/eda/box_sample.csv` (§11) |
| `configs/project.yaml` (split), `scratch.yaml`, `evaluation.yaml` (framesets, score floor) | `outputs/masks/<match>_pitch.png` (§10), `outputs/backgrounds/<match>.png`, `outputs/stages/...` |
| Oracle detections `outputs/detections/gt_oracle|gt_noisy/...` (Member 3) | `data/processed/patches/patches_<split>.npz` (§6) |
| | `outputs/scores/<detector>/<split>/patches.csv` + meta (§6.1) → E1 |
| | `outputs/proposals/<detector>/<split>/<match>__<frameset>.csv` + meta (§5) → E2 |
| | `outputs/detections/<scratch_*>/<split>/<match>__<frameset>.csv` + meta (§4) → E3, trackers |
| | `outputs/tracks/<detector>+iou_hungarian/<split>/<match>__<frameset>.csv` + meta (§3) → E4 |
| | `outputs/models/scratch_hog_svm/`, `outputs/models/scratch_cnn/` |

## 7. Data access
- Configure `configs/data_sources.yaml` from the template (see `SETUP_AND_DATA.md`). On the data machine:
  - train/val files are read **in place** from `F:\Football-Fatigue-Project\...` (read-only);
  - the validation clip 117093 is at `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4`.
- Train = 118575, 118576, 118577, 118578, 128058 (117092 optional, off). Val = 117093. Test = 128057, 132831 (frozen runs only).
- On your own machine without F:, develop with fixtures (Mode 1), or, after approval, download your own copies of only the files you need (val clip + labels ≈ 0.31 GB; five train clips + labels ≈ 1.47 GB) under `data/raw/` (Mode 2, `SETUP_AND_DATA.md` §2–3).
- Never copy, move or modify anything under `F:\`. Never duplicate clips inside the repository.

## 8. Dependencies and setup
- Python 3.12, `requirements.txt` (NumPy, SciPy, pandas, scikit-learn, OpenCV, PyTorch, torchvision, PyYAML). No new dependency without team approval; HOG is implemented by you in NumPy (scikit-image is not used).
- Install with `pip install -e . --no-deps` so `import soccertrack` works.

## 9. Milestones (dates are targets)

| ID | Target | Deliverable | Acceptance |
|---|---|---|---|
| **M1** | Fri 20, 12:00 | `data_io` core (`read_mot_raw`, `mot_to_tracks`, `load_gt_tracks`, `video_info`, `iter_frames`, `count_decoded_frames`) + `split.py` | `test_data_io.py`, `test_split.py` pass on fixtures and a tiny synthetic video (no dataset needed); foundation PR merged |
| **M2** | Fri 20 | EDA statistics for all available matches | `stats.json` matches the §11 structure; totals reported vs the paper |
| **M3** | Fri 20 (evening) | Segmentation, proposals, size model, stage images, pitch mask, **MWB-0** detections on val `eval10` | Proposal recall@0.5 and @0.3 on val reported; MWB-0 E3 run by Member 3's evaluator |
| **M4** | Sat 21 | Patch sets train/val | `validate_patch_set` passes; counts table; 1:3 ratio |
| **M5** | Sat 21 | HOG (NumPy) + Linear SVM = **MWB-1**; patch scores for val | E1 on val (CM, ROC, PR); `C` and threshold chosen on val |
| **M6** | Sat 21 → Sun 22 | Small CNN | E1 on val; learning curves; < 100k parameters |
| **M7** | Sun 22 | End-to-end scratch detector (`scratch_hog_svm`, `scratch_cnn`), scored proposals | E2/E3 on val; beats MWB-0 on val F1 or the shortfall is explained |
| **M8** *(should)* | Sun 22 | IoU + Hungarian tracker | `win60s` on `gt_oracle` (few or no ID switches), `gt_noisy`, then the best scratch detector; E4 on val |
| **M9** | Mon 23 | `operating_threshold` in `scratch.yaml`; frozen test artefacts | One logged test run; section text written |

## 10. Commands (PowerShell, repository root, `.venv` active)
```powershell
pytest tests/test_data_io.py tests/test_split.py
python scripts/run_scratch.py eda
python scripts/run_scratch.py segment --split val
python scripts/run_scratch.py detect --detector scratch_mwb0 --split val --frameset eval10
python scripts/run_scratch.py patches --split train
python scripts/run_scratch.py patches --split val
python scripts/run_scratch.py train-svm
python scripts/run_scratch.py train-cnn
python scripts/run_scratch.py score-patches --detector scratch_hog_svm --split val
python scripts/run_scratch.py detect --detector scratch_hog_svm --split val --frameset eval10
python scripts/run_scratch.py detect --detector scratch_hog_svm --split val --frameset win60s
python scripts/run_scratch.py track  --detector gt_oracle --split val --frameset win60s
python scripts/run_scratch.py track  --detector scratch_hog_svm --split val --frameset win60s
# Evaluation is run with Member 3's runner, e.g.:
python scripts/run_evaluation.py detection --detector scratch_hog_svm --split val --frameset eval10
```

## 11. Tests and acceptance
- Your test files contain one `skip` per acceptance criterion. Replace each with a real test; never delete a criterion without team agreement.
- Contract tests: every artefact you write passes `contracts.validate_*` (call `write_table` / `write_run_meta`).
- Key fixture expectations (`SHARED_CONTRACTS.md` §15):
  - `mot_raw_sample.txt` → exactly `tracks_gt.csv`;
  - `mot_raw_dirty.txt` → 11 rows and issues {1, 1, 1, 1}.
- `pytest` must pass before every push.

## 12. Working independently with fixtures
- Synthetic frames: `conftest.synthetic_frame(boxes)`. Use the boxes of `tracks_gt.csv` frame 0 to test segmentation → proposals end to end.
- Synthetic patches: `conftest.synthetic_patches()` for HOG/SVM/CNN tests.
- Tracker development: use `tracks_gt.csv` with `track_id` removed as `gt_oracle` input; perturb it yourself in tests for noise.
- Before `contracts.py` exists, follow `SHARED_CONTRACTS.md` exactly and rely on `tests/test_contracts.py` header checks.

## 13. Expected results and deliverables
- **Indicative only:** classical pipelines on tiny players typically reach clearly lower F1 than YOLO. Merged blobs in crowds and unlabelled people (referees, staff) are expected failure modes, and explaining them is part of the result.
- Deliverables:
  - artefacts in Section 6;
  - your report section (method, settings, results by `run_id`, limitations);
  - one segmentation stage example frame list for Member 4.

## 14. Integration requirements
- Artefact names, columns, frame indices and box coordinates exactly as in `SHARED_CONTRACTS.md`.
- All parameters are read from `configs/scratch.yaml`; the frameset comes from `configs/evaluation.yaml`.
- Write detections with every score ≥ `score_floor`; set `operating_threshold` in the metadata from validation.
- Foundation milestone M1 is merged first. Other members depend on it.

## 15. Common mistakes and prohibited changes
- Fitting normalisation statistics, the size model or thresholds on validation **and** test (only train is used for fitting; val for choosing).
- Building test patches or touching test clips outside the frozen run.
- Off-by-one frame errors (adding 1 anywhere outside TrackEval export), timestamp seeking, writing half-resolution boxes.
- Accuracy as the main metric on imbalanced data.
- Filtering detections at the operating threshold before writing.
- Editing `contracts.py`, `config.py`, `evaluate.py`, notebooks or other members' configs; importing ByteTrack.
- Committing `data/`, `outputs/`, `.npz`, models or `configs/data_sources.yaml`.

## 16. Definition of done
- All your acceptance tests implemented and passing.
- Every artefact validated.
- E1 (HOG+SVM, CNN), E2 and E3 (scratch detectors) Metrics JSON exist for val and for the frozen test run.
- The tracker (if completed) has E4 on oracle and detections.
- Report section written.
- Pull request merged without touching non-owned files.

## 17. Phase 3 extensions
- Read Member 3's validation failure analysis first; choose improvements by evidence.
- Add `scratch_hog_svm_v2` / `scratch_cnn_v2`:
  - hard-negative rounds (`svm.hard_negative_rounds`);
  - merged-blob splitting / perspective-scaled sliding windows (HOG+SVM) in crowded regions;
  - CNN hyper-parameter search on validation.
- Add `iou_hungarian_v2`: Kalman constant-velocity motion (`tracker_iou.use_kalman: true`), two-stage association, gating relaxed for small boxes, short-gap stitching; full-clip E4 on validation.
- Keep the Phase 2 detector and tracker IDs runnable so the Phase 2 → Phase 3 Δ table can be reproduced.
