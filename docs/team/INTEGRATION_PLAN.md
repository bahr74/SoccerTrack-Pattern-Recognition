# Integration and Ownership Plan

**Applies to:** Phase 2 (due Wed 25 Nov 2026) and Phase 3 (due Sat 19 Dec 2026).
**Read with:** [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md), [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md), and the plan [`../planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1).
**Owner of this document:** Member 4. Changes need the agreement of all members.

---

## 0. Team execution sequence

| Step | What | Who | Gate before the next step |
|---|---|---|---|
| 1 | Approve and publish the repository skeleton to `main` (PR `setup/skeleton`) | Lead + team review | Skeleton merged |
| 2 | Invite the other three members as collaborators; protect `main` | Repository owner | Invitations accepted |
| 3 | Each member clones `main`, reads their guide §0, creates their feature branch; environment installed after approval | Everyone | `pytest` runs (skips expected) |
| 4 | **Parallel foundation work** (no real data needed): Member 1 `data_io.py` + `split.py`; Member 2 tiling, mapping, NMS; Member 3 `contracts.py` + IoU; Member 4 `config.py` + data-check and setup verification | All four in parallel | Each milestone's skips replaced by passing tests |
| 5 | **Merge the foundation milestones early** (target Fri 20 Nov, 13:00); everyone pulls `main` into their branch | Reviewers + owners | Unit tests pass on `main`; data check on the lead machine |
| 6 | Members continue their own pipeline, evaluation and visualisation milestones, using fixtures and mocks wherever a dependency is not merged yet | Everyone | One PR per milestone |
| 7 | Integrate on `main`, run the smoke test (§5.1) and the validation experiments, and produce official validation results on the lead machine | Lead machine operator + owners | Outputs validate; Metrics JSON produced |
| 8 | **Freeze the configuration**, then run the held-out test evaluation once (§6) | Team decision; lead machine | One logged frozen run per system |
| 9 | Notebook, report and slides for Phase 2 (deadline Wed 25 Nov 2026) | Member 4 with all sections | Phase 2 checklist (§7) |
| 10 | Phase 3 extensions on the same contracts (deadline Sat 19 Dec 2026) | Everyone | Phase 3 checklist (§8) |

- **Approvals:** installation, any download and every Git write to `main` need the team's explicit approval.
- **Parallel work is the default**, but nobody may skip an integration checkpoint (step 5, the smoke test, the configuration freeze).
- **Code reaches the lead machine only through merged pull requests.** Official numbers are produced there from `main` (`SETUP_AND_DATA.md` §2, Mode 3).

---

## 1. Roles

| Member | Role | Branch |
|---|---|---|
| **Member 1** | Data processing and the complete from-scratch pipeline (Pipeline A: Approach A detector → IoU + Hungarian tracker) | `feature/scratch-pipeline` |
| **Member 2** | Complete pretrained YOLO pipeline (Pipeline B: tiled YOLOv8n → ByteTrack) | `feature/yolo-pipeline` |
| **Member 3** | Evaluation, contracts and model comparison | `feature/evaluation` |
| **Member 4** | Configuration, visualisation, documentation, integration; Phase 3 pitch mapping and workload | `feature/visualization-docs` |

There are **two separate tracking implementations**:
- `tracker_iou.py` (Member 1, Pipeline A)
- `tracker_bytetrack.py` (Member 2, Pipeline B)

Neither imports the other. Both consume the Detection table and emit the Track table. Both are evaluated by the same `evaluate.py` (Member 3).

---

## 2. File ownership matrix

Every file has exactly **one primary owner**. Only the owner edits it. Others may **read and import** it. Requests for changes go through the owner (Section 9).

### 2.1 Production code (`src/soccertrack/`)

| File | Owner | Main consumers (read/import only) | Notes |
|---|---|---|---|
| `__init__.py` | Member 4 | all | Version string only |
| `config.py` | **Member 4** | all | Foundation (M0). Configuration loading, path resolution, framesets, artefact paths, config hash, seeds |
| `contracts.py` | **Member 3** | all | Foundation (M0). Schemas, ID vocabulary, validation, table I/O |
| `data_io.py` | **Member 1** | Members 2, 3, 4 | Foundation (M1): `read_mot_raw`, `mot_to_tracks`, `load_gt_tracks`, `video_info`, `iter_frames` |
| `split.py` | **Member 1** | Members 2, 3 | Foundation (M1): split lists, leakage guards, `require_test_access`, `log_test_access` |
| `segmentation.py` | Member 1 | — | Produces the pitch-mask artefact (optionally used by Member 2) |
| `proposals.py` | Member 1 | — | |
| `patches.py` | Member 1 | — | |
| `features.py` | Member 1 | Member 3 (Phase 3, HOG glyphs) | |
| `classifiers.py` | **Member 1 (sole owner)** | Member 3 (E1 scores via Member 1's outputs; Phase 3 Grad-CAM via the public attributes `model`, `gradcam_layer`, `preprocess`, `hog_weights()`) | Member 3 never edits it. Interface changes need Member 3's agreement first |
| `detector_scratch.py` | Member 1 | — | |
| `tracker_iou.py` | Member 1 | — | Pipeline A tracker |
| `detector_yolo.py` | Member 2 | — | |
| `tracker_bytetrack.py` | Member 2 | — | Pipeline B tracker |
| `metrics.py` | **Member 3** | Member 1 may import `iou_matrix` | Pure functions |
| `evaluate.py` | **Member 3** | Members 1, 2 (oracle detections via files) | E1–E4, oracle inputs, TrackEval adapter. Never imports detector/tracker modules |
| `compare.py` | Member 3 | Member 4 (reads its CSV/PNG outputs) | Tables and evaluation figures |
| `explain.py` | Member 3 | — | Phase 3 |
| `visualize.py` | **Member 4** | — | Overlays, EDA and stage figures, trajectories/heatmaps. No metric computation |
| `pitch.py` | Member 4 | — | Phase 3 |
| `workload.py` | Member 4 | — | Phase 3 |

### 2.2 Everything else

| Path | Owner |
|---|---|
| `scripts/run_scratch.py` | Member 1 |
| `scripts/run_yolo.py` | Member 2 |
| `scripts/run_evaluation.py` | Member 3 |
| `scripts/run_visualization.py`, `scripts/check_data.py`, `scripts/download_test_matches.py` | Member 4 |
| `configs/scratch.yaml` | Member 1 |
| `configs/yolo.yaml` | Member 2 |
| `configs/evaluation.yaml` | Member 3 |
| `configs/project.yaml`, `configs/data_sources.example.yaml` | Member 4 (split values frozen by the plan) |
| `configs/data_sources.yaml` | each member locally (git-ignored) |
| `tests/test_data_io.py`, `test_split.py`, `test_segmentation_proposals.py`, `test_patches_features.py`, `test_classifiers.py`, `test_detector_scratch.py`, `test_tracker_iou.py` | Member 1 |
| `tests/test_detector_yolo.py`, `test_tracker_bytetrack.py` | Member 2 |
| `tests/conftest.py`, `tests/fixtures/**`, `test_contracts.py`, `test_metrics.py`, `test_evaluate.py`, `test_explain.py` | Member 3 |
| `tests/test_config.py`, `test_environment.py`, `test_visualize.py`, `test_pitch_workload.py` | Member 4 |
| `notebooks/phase2_baseline.ipynb`, `notebooks/phase3_final.ipynb` | **Member 4 (only Member 4 edits notebooks)** |
| `docs/phase2/sections/member_<n>_*.md`, `docs/phase3/sections/member_<n>_*.md` | Member n |
| `docs/phase2/**`, `docs/phase3/**` (all other files: report, slides, figures, READMEs) | Member 4 |
| `docs/team/SHARED_CONTRACTS.md` | Member 3 (changes need approval from all four members) |
| `docs/team/*` (other files, incl. member guides and prompts) | Member 4 (changes need team agreement) |
| `docs/planning/*` | Frozen baseline; edits only by team decision, applied by Member 4 |
| `docs/phase1/**` | Frozen; nobody edits |
| `README.md`, `requirements.txt`, `pyproject.toml`, `.gitignore` | Member 4 |

**Notebooks:** they are the most conflict-prone files (JSON). Only Member 4 edits them. Other members deliver Markdown text in `docs/phase<k>/sections/` and artefacts in `outputs/`. Member 4 pastes and links them.

---

## 3. Dependency graph

```
                     ┌────────────── Foundation (Fri 20 Nov, by 13:00) ──────────────┐
                     │ config.py (M4·M0)   contracts.py + metrics.iou_matrix (M3·M0)  │
                     │ data_io core + split.py (M1·M1)                                │
                     └───────┬───────────────────┬───────────────────┬───────────────┘
                             │                   │                   │
   Member 1 (Pipeline A)     │  Member 2 (B)     │  Member 3 (eval)  │  Member 4 (viz/docs)
   EDA stats (M2) ───────────┼───────────────────┼───────────────────┼──► EDA figures
   segmentation/proposals ──►│ pitch mask PNG ──►│ (optional filter) │
   MWB-0 detections ─────────┼───────────────────┼──► E3 ───────────►│ tables → notebook
   patches → HOG+SVM / CNN ──┼───────────────────┼──► E1/E2          │
   scratch detections ───────┼───────────────────┼──► E3             │ overlays
                             │ YOLO detections ──┼──► E3             │
   IoU tracker ◄── oracle ───┼── oracle ◄────────┤ gt_oracle/gt_noisy│
   tracks A ─────────────────┼── tracks B ───────┼──► E4 ───────────►│ overlays A vs B
                             └───────────────────┴───► compare ─────►│ report, slides
```

### 3.1 Unavoidable dependencies and their workarounds

| Consumer needs | Provider (milestone) | Until it is ready, use |
|---|---|---|
| Any module needs config loading | Member 4 `config.py` (M0) | Pass dicts loaded from the YAML files directly in tests; signatures are already fixed in the skeleton |
| Any producer validates output | Member 3 `contracts.py` (M0) | `SHARED_CONTRACTS.md` + the fixture-header tests in `tests/test_contracts.py` (already runnable) |
| Members 2, 3, 4 need frames / GT on real data | Member 1 `data_io` (M1) | `conftest.synthetic_frame`, `tests/fixtures/tracks_gt.csv`, `mot_raw_sample.txt` |
| Members 1, 2 tune thresholds on validation | Member 3 E3/E1 (M1) | Proposal recall (Member 1 computes it) and visual checks; the earlier reference range for YOLO |
| Trackers need oracle detections | Member 3 `gt_as_detections` / `noisy_gt_detections` (M2) | `tests/fixtures/tracks_gt.csv` without `track_id` |
| Member 2 pitch filter | Member 1 pitch-mask PNG (M3) | Run with `pitch_filter: false` (default); the metadata notes record the state |
| Member 4 EDA figures | Member 1 `outputs/eda/stats.json` (M2) | `tests/fixtures/eda_stats_sample.json` |
| Member 4 overlays / Member 3 evaluation | detector and tracker outputs | `detections_sample.csv`, `tracks_pred.csv`, `tracks_gt.csv` |
| Member 4 Phase 3 workload | PitchTrack from own `pitch.py` | `pitch_tracks_sample.csv` |

No member waits on another member's **algorithm** to develop and unit-test their own work. Real-data integration follows the order in Section 4.

---

## 4. Integration order (Phase 2)

| # | When (target) | Merge into `main` | Gate |
|---|---|---|---|
| 0 | Thu 19 Nov | Repository skeleton (this preparation) | Team review + approval |
| 1 | Fri 20 Nov, 13:00 | `config.py` (M4), `contracts.py` + `iou_matrix` (M3), `data_io` core + `split.py` (M1) | Their unit tests pass; `scripts/check_data.py` passes on the data machine |
| 2 | Fri 20 Nov, evening | EDA stats (M1); E3 + E1 evaluators (M3); EDA figures (M4) | Fixture expectations met (Contracts §15) |
| 3 | Fri 20 → Sat 21 Nov | MWB-0 detections on val (M1); tiled YOLOv8n on val (M2) | Outputs pass `contracts.validate_*`; E3 Metrics JSON produced |
| 4 | Sat 21 Nov | Patches, HOG+SVM, CNN + E1 (M1, M3); oracle inputs (M3) | E1 confusion matrices on val |
| 5 | Sun 22 Nov | Scratch detector E2/E3 (M1); IoU tracker and ByteTrack on oracle, then on detections (M1, M2); E4 (M3); overlays (M4) | Oracle sanity: GT→GT HOTA = 1.0; trackers on `gt_oracle` give few or no ID switches |
| 6 | Mon 23 Nov, by 12:00 | Operating thresholds set in `scratch.yaml` and `yolo.yaml` (from val); **configuration freeze** | All tests pass on `main`; `config_sha256` recorded |
| 7 | Mon 23 Nov, afternoon | **Frozen test run** on the data machine (Section 6) | One logged run per system |
| 8 | Mon 23 Nov, 18:00 | **Cut line:** unfinished should-haves are dropped | — |
| 9 | Tue 24 Nov | Comparison tables (M3); notebook, report, slides (M4) | Notebook executes in fast mode |
| 10 | Wed 25 Nov | Final checks and submission (M4) | Phase 2 checklist (Section 7) |

**Official runs machine:** all reported numbers are produced on **one machine**: the machine that holds `F:\Football-Fatigue-Project`, the test download and the GTX 1650 Ti. The run there uses the merged `main`. Members develop on their own machines with fixtures and, optionally, with their own downloaded clips (see `SETUP_AND_DATA.md`).

---

## 5. Independent testing strategy

- **A green `pytest` run with skipped placeholders proves nothing about the algorithms.** Skips are acceptance criteria still to be written. A milestone is complete only when its skips are replaced by real, passing tests (`pytest -rs` lists the remaining skips with their milestone IDs).

| Level | Command (PowerShell, repository root, `.venv` active) | Needs data? | Who |
|---|---|---|---|
| Contract and unit tests | `pytest` | No | everyone, before every push |
| One member's tests | `pytest tests/test_metrics.py tests/test_evaluate.py` | No | owner |
| Real-data tests | `pytest -m data` | Yes (`configs/data_sources.yaml`) | data machine |
| GPU tests | `pytest -m gpu` | GPU | Member 2, data machine |
| Fixture evaluation | `python scripts/run_evaluation.py fixtures` | No | Member 3, everyone after merges |
| Pipeline smoke test | Section 5.1 | Yes | Member 4 on the data machine after each integration step |

### 5.1 Pipeline smoke test (data machine, about 10 minutes, frameset `smoke` = frames 0–99 of 117093)

```powershell
python scripts/check_data.py --split val
python scripts/run_scratch.py detect  --detector scratch_mwb0  --split val --match 117093 --frameset smoke
python scripts/run_yolo.py    detect  --detector yolov8n_coco  --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py oracle --split val --match 117093 --frameset smoke
python scripts/run_scratch.py track   --detector gt_oracle     --split val --match 117093 --frameset smoke
python scripts/run_yolo.py    track   --detector gt_oracle     --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py detection --detector scratch_mwb0 --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py detection --detector yolov8n_coco --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py tracking  --pipeline gt_oracle+iou_hungarian --split val --match 117093 --frameset smoke
python scripts/run_evaluation.py tracking  --pipeline gt_oracle+bytetrack     --split val --match 117093 --frameset smoke
python scripts/run_visualization.py overlays --pipeline gt_oracle+bytetrack   --split val --match 117093 --frameset smoke
```

Pass criteria:
- every command exits with code 0;
- every artefact passes validation;
- every Metrics JSON passes `validate_metrics`;
- the overlay shows boxes on the players.

Smoke results are never reported.

### 5.2 Data-format compatibility checks
- Every producer calls `contracts.write_table` and `contracts.write_run_meta`, which validate before writing.
- Every consumer calls `contracts.read_table` and `contracts.read_run_meta`, which validate after reading.
- `tests/test_contracts.py` pins the fixtures to the schema constants.
- Before merging a producer branch, the owner runs the producer on the `smoke` frameset and validates the outputs.
- Box-convention check: overlays of GT (green) and predictions (other colour) on the same frame must align. Member 4 reviews one overlay per new producer.
- Frame-index check: for a detector, the first processed frame's boxes must overlap the GT of the **same** frame index (catches off-by-one and timestamp-seek errors).

---

## 6. Frozen test-run procedure (Phase 2 and Phase 3 final)

1. Each detector owner sets `operating_threshold` in their own config (chosen on validation) and merges it.
2. Member 4 confirms all tests pass on `main`. Member 3 records `config_sha256` and model hashes in `outputs/test_log.md` **before** the run.
3. On the data machine, with `main` checked out:
   - Member 1 runs `run_scratch.py patches|score-patches|detect|track --split test --frozen-test`.
   - Member 2 runs `run_yolo.py detect|track --split test --frozen-test`.
   - Member 3 runs `run_evaluation.py ... --split test --frozen-test` and `compare`.
4. No parameter, threshold or code changes after the run. If a bug invalidates the run, document it in `test_log.md`, fix it, and re-run **once**.
5. Phase 3 analysis that drives improvements uses only train/validation data.

---

## 7. Phase 2 acceptance checklist

- [ ] `pytest` passes on `main` (no failures; skips only for explicitly optional items).
- [ ] `scripts/check_data.py` passes for all train, val and test matches on the data machine.
- [ ] EDA statistics and figures, including the 0-based / −1 findings and the paper comparison.
- [ ] Split table and leakage tests; `test_log.md` with exactly one Phase 2 test entry per system.
- [ ] Segmentation stage figure and proposal recall (val).
- [ ] MWB-0, HOG+SVM and Small CNN: E1 confusion matrices (val, test).
- [ ] Approach A detector: E2 and E3 (val, test).
- [ ] Approach B (YOLOv8n tiled): E3 + FPS (val, test).
- [ ] Should-have: Pipeline A and Pipeline B tracking, E4 on oracle and on detections (val, 60-s test window).
- [ ] Comparison tables: A vs B; reproduced dataset statistics; Challenge reference with caveats; GS-HOTA marked non-comparable.
- [ ] Notebook executes in fast mode; outputs cleared in the committed version.
- [ ] Report and slides follow the assignment order; every number traceable to `outputs/metrics` or `outputs/tables`.
- [ ] No data, weights, outputs or `configs/data_sources.yaml` committed; no credentials anywhere.

## 8. Phase 3 extension checklist

- [ ] Failure analysis on train/val (Member 3), with a ranked list that justifies the improvements.
- [ ] At least one improvement with validation evidence:
  - YOLO fine-tuning `yolov8n_ft`/`yolov8s_ft` (Member 2);
  - `scratch_*_v2` (Member 1);
  - `iou_hungarian_v2` / `bytetrack_tuned` (Members 1, 2).
- [ ] New method and tracker IDs registered in `contracts.py` (already reserved).
- [ ] Explainability: Grad-CAM, SVM weight maps, permutation importance (Member 3).
- [ ] Full-clip tracking E4 on validation, then the final frozen test (second and last test access).
- [ ] Pitch mapping with leave-one-out error; workload summary from GT tracks and predicted tracks; heatmaps (Member 4).
- [ ] Final notebook, report and slides; limitations; no physiological claims.

---

## 9. Rules for conflicts between members

1. **Ownership first:** never edit a file you do not own, even for a one-line fix. Open an issue or message the owner with the exact change; the owner applies it.
2. **Contracts are frozen:** a contract change needs an issue describing the problem, a proposal by Member 3, approval from all four members, a `SCHEMA_VERSION` bump, and updated fixtures in the same pull request.
3. **Shared configuration:** each config file has one owner (Section 2.2). To change another member's parameter, ask the owner. `project.yaml` split values need approval from all members.
4. **Interface disagreements** (e.g. what `classifiers.py` exposes): the owner of the providing module decides, after the consumer has explained the need. If two members cannot agree within a day, the team decides by majority at the next sync. The plan (Revision 3.1) is the tiebreaker.
5. **Merge conflicts in owned files** are resolved by the file owner. Conflicts in shared docs are resolved by Member 4.
6. **Ambiguity:** if a contract is ambiguous, stop and ask in the team channel. Do not invent a private convention. Record the answer as a clarification in `SHARED_CONTRACTS.md` (Member 3).
7. **Daily sync (10 minutes):** each member reports merged milestones, blockers, and any requested changes to others' files.
