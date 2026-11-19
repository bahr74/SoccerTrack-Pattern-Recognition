# AI implementation prompt — Member 1 (Data + From-Scratch Pipeline A)

> Paste everything below the line into Claude Code, Codex or another coding assistant, opened in the repository root `D:\pattern_prj` (or your clone) on the branch `feature/scratch-pipeline`.

---

You are implementing **Member 1's** part of a university Pattern Recognition project (SBES 426, Medical Pattern Recognition): *Vision-Based Football Player Detection, Tracking and External Workload Analysis* on the **SoccerTrack v2** dataset (Hugging Face `atomscott/soccertrack-v2`, release v1.2, revision `325bd4c57ee8e135ab652f3921d274526ada496b`).

## Project in brief
- **Data:** fixed panoramic football clips, one 4-minute MOT clip per match, 25 fps, ~4096 × 1080 px. Ground truth (GT) has 22 player boxes per frame; median box ≈ 20 × 45 px.
- **Two complete, independent pipelines** are compared with identical evaluation:
  - **Pipeline A (you):** segmentation → region proposals → HOG + Linear SVM or Small CNN → player detection → from-scratch IoU + Hungarian tracker.
  - **Pipeline B (another member):** tiled YOLOv8n → ByteTrack.
- **Integration:** all modules communicate only through file contracts (CSV tables + JSON metadata). Evaluation is done by another member's code; you produce contract-conformant artefacts.

## FIRST SESSION (do this before writing any code)

The repository documents and `docs/team/SHARED_CONTRACTS.md` are authoritative. If this prompt and a document disagree, follow the document and tell me.

1. **Check where you are.** Run `git branch --show-current` (must be `feature/scratch-pipeline`) and `git status`. If you are on `main`, stop and tell me.
2. **Read the dataset orientation** `docs/team/SETUP_AND_DATA.md` §1 (what SoccerTrack v2 files are, 0-based frames, xywh boxes, BGR vs RGB, split and leakage rules) and §2–3 (data modes; my row). Then read `docs/team/MEMBER_1_SCRATCH.md` §0 (START HERE, dependency matrix) and the files listed below.
3. **Inspect the repository.** Open my owned skeleton files and their tests, then run `pytest -rs tests/test_data_io.py tests/test_split.py` to see the current skips. Do not change anything yet.
4. **Summarise in at most 10 lines** before coding: the milestone, the files you will touch, the skips you will replace, the shared functions you depend on and how you will work around them. Wait for my "go".
5. **First milestone:** **M1 only:** `read_mot_raw`, `mot_to_tracks`, `load_gt_tracks`, `video_info`, `count_decoded_frames`, `iter_frames` in `data_io.py`, and all of `split.py`. Replace the M1 skips in `tests/test_data_io.py` and `tests/test_split.py`.
6. **Shared functions that may still be placeholders** (they raise `NotImplementedError`): `config.load_config` / `load_data_sources` / `match_paths` (Member 4), `contracts.validate_*` / `write_table` (Member 3), `metrics.iou_matrix` (Member 3). Check with `Select-String -Path src\soccertrack\<file>.py -Pattern NotImplementedError`. Never implement, copy or edit another member's function.
7. **While a dependency is missing:** use `tests/fixtures/`, the `synthetic_frame` / `synthetic_patches` fixtures, `tmp_path` and `monkeypatch` mocks. Mark tests that need real data `@pytest.mark.data` and GPU tests `@pytest.mark.gpu`. Tell me which dependency blocks which part.
8. **Data access for this milestone:** Mode 1 (fixtures + a tiny synthetic video written by the test). Real-file checks only if my `configs/data_sources.yaml` exists (Mode 2 or the lead machine, Mode 3), via `pytest -m data`. Never download data or weights, never log in to Hugging Face, never install packages: ask me.
9. **Work only in my owned files** (list below). If something outside them must change, stop and describe the change for its owner.
10. **Stop and report after each milestone** using the handoff report at the end. Never implement several milestones or the whole of Phase 2 in one run.
11. **Report tests in three separate groups:**
    - implemented tests that ran (passed/failed);
    - placeholder tests still skipped (`pytest -rs`);
    - integration, `data` or `gpu` tests: run or not run, and why.
    A green run with skips is **not** evidence that anything is implemented.

## Read these files first and follow them exactly
0. `docs/team/SETUP_AND_DATA.md` §1–3 (dataset orientation and data modes): **before implementing any module**
1. `docs/planning/phase2_phase3_plan.md` (Revision 3.1, the technical baseline)
2. `docs/team/SHARED_CONTRACTS.md` (frozen data contracts, frame and box conventions, file naming, fixtures)
3. `docs/team/MEMBER_1_SCRATCH.md` (your task guide: scope, algorithms, milestones, acceptance criteria)
4. `docs/team/INTEGRATION_PLAN.md` (ownership matrix, dependencies, integration order)
5. `docs/team/SETUP_AND_DATA.md` and `docs/team/GIT_WORKFLOW.md`
6. The skeletons you will fill in: `src/soccertrack/{data_io,split,segmentation,proposals,patches,features,classifiers,detector_scratch,tracker_iou}.py`, `scripts/run_scratch.py`, `configs/scratch.yaml`, and your test files.

## Files you may create or edit (nothing else)
- `src/soccertrack/`: `data_io.py`, `split.py`, `segmentation.py`, `proposals.py`, `patches.py`, `features.py`, `classifiers.py`, `detector_scratch.py`, `tracker_iou.py`
- `scripts/run_scratch.py`
- `configs/scratch.yaml`
- `tests/test_data_io.py`, `tests/test_split.py`, `tests/test_segmentation_proposals.py`, `tests/test_patches_features.py`, `tests/test_classifiers.py`, `tests/test_detector_scratch.py`, `tests/test_tracker_iou.py`
- `docs/phase2/sections/member_1_scratch.md`, `docs/phase3/sections/member_1_scratch.md`

You may add **private helper functions** inside these files. Do **not** create new modules without asking me.

## Files you must NOT modify
- **Other members' code:** `config.py`, `contracts.py`, `metrics.py`, `evaluate.py`, `compare.py`, `explain.py`, `detector_yolo.py`, `tracker_bytetrack.py`, `visualize.py`, `pitch.py`, `workload.py`.
- **Others' configs and tests:** other `configs/*.yaml`, `tests/conftest.py`, `tests/fixtures/**`.
- **Shared files:** notebooks, `docs/team/**`, `docs/planning/**`, `docs/phase1/**`, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`.
- **Data:** anything under `F:\`.

If you believe one of these needs a change, **stop and tell me** the exact change and why. I will ask the owner.

## Interfaces you use (read/import only)
- `soccertrack.config`: `load_config`, `load_data_sources`, `match_paths`, `frameset_range`, `artifact_path`, `config_sha256`, `set_seeds` (Member 4).
- `soccertrack.contracts`: `write_table`, `read_table`, `write_run_meta`, `read_run_meta`, `validate_*`, schema constants (Member 3).
- `soccertrack.metrics.iou_matrix` (Member 3; optional).

If one of these is not implemented yet:
- in tests, use the committed fixtures and plain pandas/NumPy;
- keep your production code calling the official interface;
- tell me which dependency is missing.

**Never** import `tracker_bytetrack`, `detector_yolo` or anything from Pipeline B.

## Conventions that are not negotiable (from SHARED_CONTRACTS.md)
- **Frames are 0-based.** Frame k = k-th frame from **sequential** decoding (`cv2.VideoCapture.grab()`/`retrieve()`). Never seek by time; clip 117093 has a 0.979 s container offset.
- **Raw MOT files** have 10 columns: `frame,id,x,y,w,h,-1,-1,-1,-1` (verified on the real files). Convert to the Track table `frame,track_id,x,y,w,h,score` with `score = 1.0`.
- **Boxes:** top-left `x, y` and `w, h` in **full-resolution** pixels of the original frame. If you compute at `work_scale = 0.5`, rescale before writing.
- **Images:** frames are BGR `uint8` (OpenCV); patches are **RGB** `uint8` 64 (h) × 32 (w).
- **Detection tables** contain **all** detections with `score ≥ score_floor` (0.05, from `configs/evaluation.yaml`) after NMS (`torchvision.ops.nms`). The operating threshold is chosen on validation and written into the metadata, not applied to the file.
- **File names:**
  - detections: `outputs/detections/<detector>/<split>/<match>__<frameset>.csv` + `.meta.json`
  - proposals: `outputs/proposals/...`, same pattern
  - tracks: `outputs/tracks/<detector>+iou_hungarian/<split>/<match>__<frameset>.csv` + `.meta.json`
  - Use `config.artifact_path`.
- **IDs and framesets:**
  - detector IDs: `scratch_mwb0`, `scratch_hog_svm`, `scratch_cnn`; tracker ID: `iou_hungarian`;
  - framesets come from `configs/evaluation.yaml` (`eval10`, `win60s`, `full`, `smoke`). Do not invent frame selections.

## Algorithms to implement (details in MEMBER_1_SCRATCH.md §3)
1. **MOT loading and cleaning:**
   - `read_mot_raw`, `mot_to_tracks`: drop exact duplicates and degenerate boxes (w ≤ 1 or h ≤ 1); clip to the image; count `frames_not_expected`.
   - `load_gt_tracks`, `video_info`, `count_decoded_frames`, `iter_frames`.
   - `compute_eda_stats` producing exactly the §11 JSON structure.
2. **Split:** `matches_for`, `split_of`, `assert_disjoint`, `require_test_access` (raise unless `--frozen-test`), `log_test_access` (append to `outputs/test_log.md` with `config_sha256`).
3. **Segmentation:**
   - median background of `background_num_frames` frames spread over the clip;
   - HSV pitch mask with the largest component + hole filling (fallbacks: keypoint polygon after verifying pixel space; manual polygon from config);
   - foreground = blurred Lab/gray absolute difference > threshold inside the mask;
   - median filter, opening, closing, small vertical dilation.
   - Save the full-resolution pitch mask PNG and stage images.
4. **Proposals:**
   - `cv2.connectedComponentsWithStats` → boxes (full-resolution);
   - `SizeModel` h(y_bottom) = a·y + b fitted by least squares on **train** GT only;
   - keep blobs within `size_tolerance`; split blobs wider than `split_width_factor` × expected width;
   - `proposal_recall`.
5. **MWB-0:** `rule_scores` from aspect ratio, fill ratio and size-model agreement → NMS → Detection table (`scratch_mwb0`).
6. **Patches:** `crop_patch` (10 % context, zero-pad at borders, BGR→RGB, resize to 64 × 32); `build_patch_set` with positives = GT and negatives = random (IoU < 0.1, inside the pitch mask) / partial (0.1 ≤ IoU < 0.4) / unmatched proposals at 1:3. Save with `numpy.savez_compressed` and the `meta_json` key.
7. **HOG from scratch in NumPy:**
   - grayscale; gradients with [−1, 0, 1]; 8 × 8 cells; 9 unsigned bins (0–180°) with bilinear voting;
   - 2 × 2-cell blocks, stride 1 cell; L2-Hys (clip 0.2) → exactly **756** features for 64 × 32.
   - Test against `cv2.HOGDescriptor((32,64),(16,16),(8,8),(8,8),9)` and document the tolerance and any ordering difference.
8. **HOG + Linear SVM:**
   - `LinearSVC(class_weight="balanced")`; choose `C` from `svm.C_grid` by **validation** PR-AUC;
   - probability calibration with `CalibratedClassifierCV` on **train** folds;
   - `hog_weights()` returns the 756-d weight vector.
9. **Small CNN (PyTorch):**
   - 3 × (Conv3×3 → BN → ReLU → MaxPool2), channels 16/32/64 → global average pool → dropout → Linear(1); < 100k parameters.
   - Weighted BCE; Adam; augmentation per config; early stopping on **validation** PR-AUC.
   - Mean/std from **train** patches only, stored with the model.
   - Public attributes `model`, `gradcam_layer` (last conv), `preprocess(X)`.
10. **Scratch detector:** proposals → classifier scores → write the scored Proposal table (before NMS) → NMS → Detection table + metadata (runtime, FPS, `operating_threshold` from `scratch.yaml`). Also write patch scores for E1 (`outputs/scores/<detector>/<split>/patches.csv`).
11. **IoU + Hungarian tracker:**
    - cost = 1 − IoU; `scipy.optimize.linear_sum_assignment`; reject pairs below `iou_gate`;
    - tentative → confirmed after `n_init` hits; delete after `max_age` misses; only confirmed tracks are output;
    - detections below `min_score` (default: the detector's operating threshold) are ignored;
    - every frame of the stride-1 frameset is processed, including empty ones.
    - Validate on `gt_oracle` first, then `gt_noisy`, then your detections.

## Milestones: implement in this order, and STOP after each one to report
- **M1:** `data_io` core + `split`. This is a foundation for other members, so merge it early.
- **M2:** EDA statistics.
- **M3:** segmentation + proposals + MWB-0 on val `eval10`.
- **M4:** patch sets (train, val).
- **M5:** HOG + SVM, patch scores (val).
- **M6:** Small CNN.
- **M7:** end-to-end detectors `scratch_hog_svm` / `scratch_cnn` (val `eval10`, `win60s`).
- **M8 (should-have):** IoU + Hungarian tracker.
- **M9:** set operating thresholds from validation; test artefacts **only** when I explicitly say the frozen test run has started.

## Data and model paths
- All paths come from `configs/data_sources.yaml` (git-ignored; template `configs/data_sources.example.yaml`).
- On the data machine, train/val files are referenced **in place, read-only**, under `F:\Football-Fatigue-Project\` (e.g. validation clip `F:\Football-Fatigue-Project\data\external\soccertrack-v2\raw\117093\117093.mp4`).
- **Split:** train = 118575, 118576, 118577, 118578, 128058 (117092 optional, disabled); val = 117093; test = 128057, 132831.
- Write only under the repository's `data/processed/` and `outputs/`.

## Hard rules
- **No test leakage.**
  - Fit anything (normalisation, size model, SVM, CNN) on **train** only; choose anything (thresholds, `C`, epochs, morphology parameters) on **val** only.
  - Never read test clips or test GT unless I give you the frozen-test instruction; then pass `--frozen-test` and log the access.
- **No downloads** of data or weights and **no package installation** without my explicit approval.
- **No copying or moving** of dataset files; never write to `F:\`.
- **No git commits, pushes or branch changes** unless I ask.
- **Code quality:**
  - readable, small, documented functions (docstrings with shapes, dtypes and units; type hints);
  - parameters from `configs/scratch.yaml` only; deterministic seeds via `config.set_seeds`;
  - no notebooks, no dead code, no new dependencies.
- **Tests:**
  - replace each `skip` in your test files with a real test using the fixtures in `tests/fixtures/` and `tests/conftest.py`;
  - tests must not need real data unless marked `@pytest.mark.data`;
  - run `pytest` before reporting.
- **Phase 3 compatibility:** keep public function names and signatures from the skeleton. Add new behaviour behind config flags (e.g. `tracker_iou.use_kalman`, `svm.hard_negative_rounds`) and new IDs (`scratch_*_v2`, `iou_hungarian_v2`). Do not break Phase 2 IDs.
- **Ambiguity:** if a contract, interface or path is unclear or seems wrong, **stop and ask me**. Do not guess or create a private convention.

## Validation commands (PowerShell, `.venv` active)
```powershell
pytest
pytest tests/test_data_io.py tests/test_split.py tests/test_segmentation_proposals.py tests/test_patches_features.py tests/test_classifiers.py tests/test_detector_scratch.py tests/test_tracker_iou.py
pytest -m data tests/test_data_io.py
python scripts/run_scratch.py detect --detector scratch_mwb0 --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py detection --detector scratch_mwb0 --split val --match 117093 --frameset smoke   # when Member 3's evaluator exists
```

## How to report after each milestone (handoff report)

Reply with exactly these sections:

```
### Handoff report — Member 1, milestone M<k>
Status: complete | partial | blocked
Changed files (owned only): <list>
Implemented: <3–6 bullets>
Tests
  - Implemented and run: <command> → <real summary: N passed, N failed>
  - Skips replaced in this milestone: <test names>
  - Skips still remaining (pytest -rs): <count, with milestone IDs>
  - Integration / data / gpu tests: <run or not run, and why>
Results: <key numbers (issue counts, frame counts) with `run_id`s where applicable>
Dependencies used / still placeholders: <function → owner → workaround used>
Deviations from the guide: <none | what and why>
Blockers / requests for other members: <none | who, what>
Next step: <the next milestone, not started>
```

Never claim tests passed without running them. Do not commit or push unless I ask; when I do, stage only owned files and show `git diff --cached --name-only` first.
