# AI implementation prompt — Member 3 (Contracts, Evaluation and Comparison)

> Paste everything below the line into Claude Code, Codex or another coding assistant, opened in the repository root `D:\pattern_prj` (or your clone) on the branch `feature/evaluation`.

---

You are implementing **Member 3's** part of a university Pattern Recognition project (SBES 426, Medical Pattern Recognition): *Vision-Based Football Player Detection, Tracking and External Workload Analysis* on **SoccerTrack v2** (release v1.2).

## Project in brief
- **Two complete, independent pipelines** detect and track the 22 players in panoramic clips:
  - **Pipeline A:** from-scratch detector + IoU + Hungarian tracker.
  - **Pipeline B:** tiled YOLOv8n + ByteTrack.
- **Your role:** you own the **data contracts** that all members rely on and the **evaluation** that compares both pipelines fairly:
  - E1 patch classification, E2 proposal classification, E3 end-to-end detection, E4 tracking with TrackEval;
  - comparison tables; in Phase 3, failure analysis and explainability evaluation.
- You never wait for a detector: you develop and test everything on the committed synthetic fixtures, which have hand-computed expected values.

## FIRST SESSION (do this before writing any code)

The repository documents and `docs/team/SHARED_CONTRACTS.md` are authoritative. If this prompt and a document disagree, follow the document and tell me.

1. **Check where you are.** Run `git branch --show-current` (must be `feature/evaluation`) and `git status`. If you are on `main`, stop and tell me.
2. **Read the dataset orientation** `docs/team/SETUP_AND_DATA.md` §1 (what SoccerTrack v2 files are, 0-based frames, xywh boxes, BGR vs RGB, split and leakage rules) and §2–3 (data modes; my row). Then read `docs/team/MEMBER_3_EVALUATION.md` §0 (START HERE, dependency matrix) and the files listed below.
3. **Inspect the repository.** Open my owned skeleton files and their tests, then run `pytest -rs tests/test_contracts.py tests/test_metrics.py` to see the current skips. Do not change anything yet.
4. **Summarise in at most 10 lines** before coding: the milestone, the files you will touch, the skips you will replace, the shared functions you depend on and how you will work around them. Wait for my "go".
5. **First milestone:** **M0 only:** `contracts.py` (schemas, all `validate_*` including the metric value domains, `read_table`/`write_table`, `read_run_meta`/`write_run_meta`, `pipeline_id`) and `metrics.iou_matrix`. Replace the M0 skips in `tests/test_contracts.py` and the IoU skip in `tests/test_metrics.py`.
6. **Shared functions that may still be placeholders** (they raise `NotImplementedError`): `data_io.load_gt_tracks` (Member 1), `config.*` (Member 4), `split.*` (Member 1). Check with `Select-String -Path src\soccertrack\<file>.py -Pattern NotImplementedError`. Never implement, copy or edit another member's function.
7. **While a dependency is missing:** use `tests/fixtures/`, the `synthetic_frame` / `synthetic_patches` fixtures, `tmp_path` and `monkeypatch` mocks. Mark tests that need real data `@pytest.mark.data` and GPU tests `@pytest.mark.gpu`. Tell me which dependency blocks which part.
8. **Data access for this milestone:** Mode 1 only. Use `tests/fixtures/`; never wait for other members' outputs. Never download data or weights, never log in to Hugging Face, never install packages: ask me.
9. **Work only in my owned files** (list below). If something outside them must change, stop and describe the change for its owner.
10. **Stop and report after each milestone** using the handoff report at the end. Never implement several milestones or the whole of Phase 2 in one run.
11. **Report tests in three separate groups:**
    - implemented tests that ran (passed/failed);
    - placeholder tests still skipped (`pytest -rs`);
    - integration, `data` or `gpu` tests: run or not run, and why.
    A green run with skips is **not** evidence that anything is implemented.

## Read these files first and follow them exactly
0. `docs/team/SETUP_AND_DATA.md` §1–3 (dataset orientation and data modes): **before implementing any module**
1. `docs/planning/phase2_phase3_plan.md` (Revision 3.1; Sections 3.5 comparability and 5 evaluation)
2. `docs/team/SHARED_CONTRACTS.md`: you own it, and `contracts.py` must implement it exactly; §15 lists the fixture expectations
3. `docs/team/MEMBER_3_EVALUATION.md` (your task guide)
4. `docs/team/INTEGRATION_PLAN.md`, `docs/team/SETUP_AND_DATA.md`, `docs/team/GIT_WORKFLOW.md`
5. Skeletons: `src/soccertrack/{contracts,metrics,evaluate,compare,explain}.py`, `scripts/run_evaluation.py`, `configs/evaluation.yaml`, `tests/conftest.py`, `tests/fixtures/`, `tests/test_{contracts,metrics,evaluate,explain}.py`

## Files you may create or edit (nothing else)
- `src/soccertrack/`: `contracts.py`, `metrics.py`, `evaluate.py`, `compare.py`, `explain.py`
- `scripts/run_evaluation.py`
- `configs/evaluation.yaml`
- `tests/conftest.py`, `tests/fixtures/**`, `tests/test_contracts.py`, `tests/test_metrics.py`, `tests/test_evaluate.py`, `tests/test_explain.py`
- `docs/team/SHARED_CONTRACTS.md`: **only** with the explicit approval of all four members (I will confirm). Clarifications that do not change behaviour can be proposed to me first.
- `docs/phase2/sections/member_3_evaluation.md`, `docs/phase3/sections/member_3_evaluation.md`

## Files you must NOT modify
- **Other members' code:** `config.py`, `data_io.py`, `split.py`, `segmentation.py`, `proposals.py`, `patches.py`, `features.py`, `classifiers.py`, `detector_scratch.py`, `tracker_iou.py`, `detector_yolo.py`, `tracker_bytetrack.py`, `visualize.py`, `pitch.py`, `workload.py`.
- **Other configs and tests.**
- **Shared files:** notebooks, `docs/planning/**`, `docs/phase1/**`, other `docs/team/**` files, `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore`.
- **Data:** anything under `F:\`.

**Never import detector or tracker modules.** You evaluate their **output files**.

## Interfaces you use (read/import only)
- `soccertrack.config`: `load_config`, `load_data_sources`, `match_paths`, `frameset_range`, `artifact_path`, `config_sha256`.
- `soccertrack.data_io.load_gt_tracks` (real data only).
- `soccertrack.split`: `matches_for`, `require_test_access`, `log_test_access`.
- **Phase 3:** `soccertrack.classifiers` public attributes `predict_proba`, `model`, `gradcam_layer`, `preprocess`, `hog_weights()`.

If one is missing, use the fixtures and report it.

## What to implement
1. **`contracts.py` (milestone M0: highest priority, others depend on it):**
   - `validate_tracks`, `validate_detections` (incl. frames inside the metadata frameset and scores ≥ `score_floor`), `validate_proposals`, `validate_patch_set`, `validate_patch_scores`, `validate_run_meta`, `validate_metrics` (task-specific keys), `validate_pitch_tracks`;
   - `read_table` / `write_table` (exact column order; boxes 2 decimals, scores 4 decimals; sorted; no index);
   - `read_run_meta` / `write_run_meta`; `pipeline_id`.
   - Raise `ContractError` with a precise message. Keep the existing constants; do not rename them.
2. **`metrics.py`:**
   - `iou_matrix` (xywh; M0);
   - `match_frame` (greedy by score, one-to-one, IoU ≥ threshold);
   - `precision_recall_f1` (0/0 → 0);
   - `average_precision` (all-point interpolated envelope over detections pooled across frames; returns the PR curve);
   - `binary_classification_metrics` (score ≥ threshold → positive; confusion matrix, sensitivity, specificity, precision, F1, accuracy, ROC-AUC, PR-AUC via `sklearn.metrics`; curves ≤ 200 points).
3. **`evaluate.py`:**
   - **E1** from Patch score tables.
   - **E2:** label scored proposals by IoU ≥ 0.5 with GT of the same frame.
   - **E3:** GT restricted to the metadata frameset; P/R/F1 at the metadata `operating_threshold` for IoU 0.5 and 0.3; AP@0.5 and AP@[.5:.95] over all detections; recall per height bin; FP per frame; FPS copied from metadata.
   - **Oracle inputs:** `gt_as_detections`, `noisy_gt_detections` (drop 10 %, σ = 2 px jitter, scores U[0.3, 1], seed from config, clipped).
   - **`export_trackeval`**, exactly per `SHARED_CONTRACTS.md` §9:
     - GT lines `frame+1−start,id,x,y,w,h,1,1,1`;
     - tracker lines `frame+1−start,id,x,y,w,h,score,-1,-1,-1`;
     - `seqinfo.ini`; folder layout as specified.
   - **`evaluate_tracking`:** run `trackeval.Evaluator` + `trackeval.datasets.MotChallenge2DBox` with `GT_FOLDER`, `TRACKERS_FOLDER`, `TRACKERS_TO_EVAL`, `SEQ_INFO`/sequence list, `SKIP_SPLIT_FOL=True`, metrics `HOTA`, `CLEAR`, `Identity`. Map the results to `hota, deta, assa, loca, idf1` (fractions in [0, 1]), `mota` (≤ 1, may be negative) and `idsw` (integer).
   - ⚠️ Verify the configuration key names against the installed trackeval 1.3.0 source; do not guess. The GT-vs-GT test must give 1.0.
   - `pool_metrics` (sum counts, recompute ratios); `write_metrics` (validated; path `outputs/metrics/<run_id>.json`; `run_id = <eval_id>__<method>__<split>__<match|pooled>__<frameset>`; provenance with `config_sha256`, package versions, seed).
4. **`compare.py`:**
   - tables: detection A vs B, classifiers, tracking incl. oracle rows, dataset statistics vs paper, reference table with comparability labels;
   - figures: confusion matrices, PR/ROC overlays, recall by height.
5. **`scripts/run_evaluation.py` commands:** `oracle`, `patches`, `proposals`, `detection`, `tracking`, `compare`, `fixtures`, (Phase 3) `failures`.
6. **Phase 3, `explain.py`:** error stratification (height bin, region, crowding, FP/FN types), ID-switch events, Grad-CAM, SVM weight maps, permutation importance by feature group. Train/val only.

## Acceptance values (from the committed fixtures; your tests must reproduce them)
- **IoU:** (500,140,18,40) vs (510,150,18,40) = 0.2.
- **`detections_sample.csv` vs `tracks_gt.csv`:**
  - IoU 0.5, threshold 0.5: TP 7, FP 1, FN 2, P 0.8750, R 0.7778, F1 0.8235;
  - all detections: TP 8, FP 3, all-point AP@0.5 = 0.8667.
- **`patch_scores_sample.csv`** at 0.5: TP 3, FN 1, TN 3, FP 1, sensitivity 0.75, specificity 0.75, ROC-AUC 0.9375.
- **`tracks_pred.csv` vs `tracks_gt.csv`:** MOTA 0.6667, IDF1 0.7500, IDSW 1. GT vs GT: HOTA = IDF1 = MOTA = 1.0.
- **`export_trackeval`** of the fixtures reproduces `tests/fixtures/trackeval_example/` byte for byte.
- **Own AP vs `pycocotools`:** within 0.01 on a random synthetic set (document the interpolation difference).

If your implementation disagrees with a value, re-derive it by hand before changing anything. If the fixture is wrong, **stop and report**; do not silently edit fixtures.

## Milestones: implement in this order and STOP after each one to report
- **M0:** `contracts.py` + `iou_matrix` (merge first).
- **M1:** metrics + E1/E2/E3 + `write_metrics` + `fixtures` command.
- **M2:** oracle detections.
- **M3:** TrackEval export + E4.
- **M4:** pooling, comparison tables and figures.
- **M5:** frozen test evaluation **only** after my explicit instruction (log it).

## Data paths
- GT txt files come through `config` + `data_io` (on the data machine read-only under `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\`).
- You need no video. Write only under `outputs/`.

## Hard rules
- **No test leakage:** you never choose thresholds; you read `operating_threshold` from each producer's metadata. Test-split evaluation only with `--frozen-test` after my instruction, logged.
- **No downloads, installs, data copies or writes to `F:\`. No git commits, pushes or branch changes** unless I ask.
- **Metric domains (SHARED_CONTRACTS.md §1):** bounded ratios (P, R, F1, AP, AUCs, HOTA, DetA, AssA, LocA, IDF1) are fractions in [0, 1]; MOTA ≤ 1 and may be negative (never clip it); FP/frame and FPS are ≥ 0 and may exceed 1; IDSW and counts are integers. `validate_metrics` enforces this. Never report GS-HOTA as comparable; never present earlier Football Fatigue numbers as our results.
- **Code quality:** pure, typed, documented metric functions; no I/O in `metrics.py`; parameters from `configs/evaluation.yaml`.
- **Tests:** replace every `skip` in your test files with real tests; also test that validators **reject** bad tables; run `pytest` before reporting.
- **Phase 3 compatibility:** do not rename public functions or constants once M0 is merged. New IDs are already reserved in `contracts.py`.
- **Ambiguity:** if any contract detail is unclear, stop and ask me. I will coordinate a team decision.

## Validation commands (PowerShell, `.venv` active)
```powershell
pytest
pytest tests/test_contracts.py tests/test_metrics.py tests/test_evaluate.py
python scripts/run_evaluation.py fixtures
python scripts/run_evaluation.py oracle --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py tracking --pipeline gt_oracle+iou_hungarian --split val --match 117093 --frameset smoke
```

## How to report after each milestone (handoff report)

Reply with exactly these sections:

```
### Handoff report — Member 3, milestone M<k>
Status: complete | partial | blocked
Changed files (owned only): <list>
Implemented: <3–6 bullets>
Tests
  - Implemented and run: <command> → <real summary: N passed, N failed>
  - Skips replaced in this milestone: <test names>
  - Skips still remaining (pytest -rs): <count, with milestone IDs>
  - Integration / data / gpu tests: <run or not run, and why>
Results: <fixture values: expected vs obtained>
Dependencies used / still placeholders: <function → owner → workaround used>
Deviations from the guide: <none | what and why>
Blockers / requests for other members: <none | who, what>
Next step: <the next milestone, not started>
```

Never claim tests passed without running them. Do not commit or push unless I ask; when I do, stage only owned files and show `git diff --cached --name-only` first.
