# Member 3 — Evaluation, Contracts and Model Comparison

Branch: `feature/evaluation` · AI prompt: [`prompts/MEMBER_3_AI_PROMPT.md`](prompts/MEMBER_3_AI_PROMPT.md)
Read first: [`../planning/phase2_phase3_plan.md`](../planning/phase2_phase3_plan.md) (Revision 3.1, Sections 3.5 and 5), [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md) (you own it), [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md), [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md), [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md).

---

## 0. START HERE — Your first working session

Your first task: **M0: complete `contracts.py` (schemas, validators, table and metadata I/O, metric value domains) and `metrics.iou_matrix`.**

Data mode for this task: **Mode 1 only.** The fixtures contain ground truth, predictions and hand-computed expected answers. You do **not** wait for models, videos or other members' outputs (modes: [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §2; your data plan: §3).

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
   4. [`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md): all of it (you own it); start with §1, §3, §4, §7, §8, §15;
   5. [`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0, §2–4;
   6. [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §1–5.
3. **Create your branch:**
   ```powershell
   git switch main; git pull origin main
   git switch -c feature/evaluation
   ```
4. **Know your files:**
   - **Owned** (edit only these): `src/soccertrack/{contracts,metrics,evaluate,compare,explain}.py`, `scripts/run_evaluation.py`, `configs/evaluation.yaml`, `tests/conftest.py`, `tests/fixtures/**`, `tests/test_{contracts,metrics,evaluate,explain}.py`, `docs/team/SHARED_CONTRACTS.md` (changes need all four members), your two `sections/` files.
   - **Read-only** (import or read, never edit): `config.py`, `data_io.py`, `split.py`, `classifiers.py` (Phase 3, public attributes only), other configs, notebooks.
   - **Off-limits:** `detector_scratch.py`, `detector_yolo.py`, `tracker_iou.py`, `tracker_bytetrack.py` (you evaluate their **files**, never import their code); `docs/phase1/**`; `F:\Football-Fatigue-Project`.
5. **Python environment** (only after the team has approved installation): [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §4.
6. **Initial fixture checks** (✅ work immediately, no data needed):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pytest                                  # a few contract/fixture checks pass; most tests are SKIPPED placeholders
   pytest -rs tests/test_contracts.py tests/test_metrics.py                 # shows the skipped acceptance tests of your first milestone
   python scripts/run_evaluation.py --help             # argument parsing only; the commands themselves are not implemented yet
   ```
   A green run with skips proves nothing about your code. Any call into an unimplemented function raises `NotImplementedError("TODO(Member n, Mk)")`.
7. **Understand your first milestone:** §9 (milestones) and §0.2–0.3 below.
8. **Implement only that milestone.** Do not start the next one in the same branch state.
9. **Replace the skipped tests of the milestone** with real tests (remove the `@pytest.mark.skip`, write the assertions):
   - `test_contracts.py`: validators accept every fixture; reject bad tables; metric value domains; detections outside the frameset; write/read round trip; unknown IDs
   - `test_metrics.py`: `iou_matrix` (identical 1, disjoint 0, (500,140,18,40) vs (510,150,18,40) = 0.2)
10. **Run and inspect:**
    ```powershell
    pytest tests/test_contracts.py tests/test_metrics.py -v
    pytest -rs                              # no skip with your milestone ID may remain
    ```
11. **Document:** docstrings in your modules, and a short note of what was done in `docs/phase2/sections/member_3_evaluation.md`.
12. **Report blockers** to the owner of the missing piece, through the team chat or the PR, never by editing their file ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §9).
13. **Commit and open a PR** (only when your team has authorised it):
    ```powershell
    git status
    git add <your owned files only>
    git diff --cached --name-only           # nothing outside your owned files; no data, weights, outputs or credentials
    git commit -m "M<k>: <what was done>"
    git push -u origin feature/evaluation
    ```
    Then open a pull request into `main` with the checklist in [`GIT_WORKFLOW.md`](GIT_WORKFLOW.md) §5.

**Commands that work only later:**

| Command | Works |
|---|---|
| `python scripts/run_evaluation.py fixtures` | ⏳ after M1 |
| `python scripts/run_evaluation.py oracle …` | ⏳ after M2 (needs Member 1's `load_gt_tracks` and real GT on a data machine) |
| `python scripts/run_evaluation.py detection|tracking …` on real outputs | ⏳ after the producing member's milestone is merged |

### 0.1 Your data
Data needed now → source → local configuration → verification → data needed later: see your row in [`SETUP_AND_DATA.md`](SETUP_AND_DATA.md) §3. Never put datasets, weights, outputs or credentials into Git.

### 0.2 Dependency and independence matrix

| Milestone | Required input | Exists now? | Owner / provider | If not ready yet, use | Proof of completion | Hand-off |
|---|---|---|---|---|---|---|
| **M0** contracts + IoU | Fixtures, `SHARED_CONTRACTS.md` | Yes (you own them) | Member 3 | Nothing missing | All `test_contracts.py` skips and the IoU skip replaced and passing | Validators, `read_table`/`write_table`, `iou_matrix` → everyone |
| **M1** E1/E2/E3, `write_metrics` | `detections_sample.csv`, `tracks_gt.csv`, `proposals_sample.csv`, `patch_scores_sample.csv` | Yes | Member 3 | Nothing missing | Fixture values reproduced (TP 7/FP 1/FN 2, AP 0.8667, ROC-AUC 0.9375) | `run_evaluation.py` detection/patch commands → Members 1, 2 |
| **M2** oracle detections | Real GT Track tables | After Member 1 M1 + Member 4 M0; real GT on a data machine | Members 1, 4 | Develop on `tracks_gt.csv`; run on val on the lead machine | Valid `gt_oracle`/`gt_noisy` tables for val `win60s` and `smoke` | Oracle Detection tables → Members 1, 2 (trackers) |
| **M3** TrackEval + E4 | `tracks_pred.csv`, `trackeval_example/` | Yes | Member 3 | Nothing missing for the logic | Byte-identical export; GT vs GT = 1.0; MOTA 0.6667, IDF1 0.75, IDSW 1 | E4 command → Members 1, 2 |
| **M4** comparison | Real Metrics JSON from the runs | After the producers' milestones | Members 1, 2 (outputs); lead machine (runs) | `metrics_sample.json` | Tables and figures generated from Metrics JSON | Comparison tables and evaluation figures → Member 4 (report) |

Parallel work is allowed, but integration checkpoints cannot be skipped. Each milestone that others depend on is merged into `main` before they rely on it ([`INTEGRATION_PLAN.md`](INTEGRATION_PLAN.md) §0 and §4).

### 0.3 Working from fixtures, then switching to real outputs
- `tests/fixtures/` holds small synthetic GT, detections, tracks, proposals and patch scores whose correct metric values were computed by hand ([`SHARED_CONTRACTS.md`](SHARED_CONTRACTS.md) §15). Your tests must reproduce them exactly.
- You can build the whole evaluation system (E1–E4, TrackEval export, pooling, comparison tables) **without** any trained model, video or prediction from Members 1 and 2.
- When real outputs arrive, they are files with exactly the same contract (`outputs/detections/…csv`, `outputs/tracks/…csv`, `.meta.json`). You only change the **input paths** given to the runner. The evaluation logic does not change. If a real file fails validation, the producer fixes it; the validator is not loosened.
- **Order after M0:** E1/E2/E3 + `write_metrics` (M1) → oracle detections (M2) → TrackEval export + E4 (M3) → pooling, comparisons and figures (M4) → frozen test evaluation (M5).

---

## 1. Project overview and scientific objective
- **Task:** detect and track the 22 players in SoccerTrack v2 panoramic clips (release v1.2). Two complete pipelines are compared:
  - **Pipeline A (Member 1):** from-scratch detector + IoU + Hungarian tracker.
  - **Pipeline B (Member 2):** tiled YOLOv8n + ByteTrack.
- **Your scientific role:** make the comparison **fair, correct, reproducible and honest**. You own:
  - the contracts that let everyone work independently;
  - the metric implementations;
  - the four evaluations (E1 patch classification, E2 proposal classification, E3 end-to-end detection, E4 tracking);
  - the comparison with the paper and the SoccerTrack Challenge;
  - in Phase 3, failure analysis and explainability evaluation.
- You **never** wait for a detector: everything is developed and tested on the synthetic fixtures with hand-computed answers.

## 2. Your scope

**Phase 2:**
1. **Contracts (M0, first):** implement `contracts.py` exactly as `SHARED_CONTRACTS.md` describes; maintain `tests/fixtures/` and `tests/conftest.py`.
2. Metrics: IoU matrix, greedy matching, P/R/F1, all-point AP@0.5 and AP@[.5:.95], binary-classification metrics (confusion matrix, sensitivity, specificity, precision, F1, accuracy, ROC-AUC, PR-AUC).
3. E1 (patch scores), E2 (scored proposals), E3 (Detection tables), E4 (Track tables via TrackEval 1.3.0: HOTA, DetA, AssA, LocA, IDF1, MOTA, IDSW).
4. Oracle inputs for both trackers: `gt_oracle` and `gt_noisy` Detection tables.
5. Pooled results; comparison tables; evaluation figures (confusion matrices, PR/ROC curves).
6. Coordination of the frozen test protocol (config hash, test log) together with the detector owners.

**Phase 3:**
- Failure analysis on **train/val** (errors by box height, image region, crowding; FP types; ID-switch events).
- Explainability evaluation: Grad-CAM on the Small CNN, SVM weight maps, permutation importance by feature group.
- Phase 2 → Phase 3 Δ tables; final frozen evaluation.

## 3. Algorithms, methods and concepts
- **IoU** on xywh boxes; **greedy matching per frame** by descending score at IoU ≥ threshold (one-to-one).
- **P/R/F1** at the method's `operating_threshold` (from the run metadata) for IoU 0.5 and 0.3; **FP per frame**; **recall by GT height bin** (`evaluation.yaml` bins).
- **AP:**
  - Pool all detections (score ≥ floor) over frames, sort by score, compute cumulative TP/FP, and integrate the all-point interpolated precision envelope.
  - AP@[.5:.95] = mean over IoU 0.50:0.05:0.95.
  - Cross-check with `pycocotools` within 0.01 on synthetic data; document any systematic difference (e.g. 101-point interpolation).
- **Classification metrics:** `score ≥ threshold` → positive; sensitivity = TP/(TP+FN); specificity = TN/(TN+FP); ROC-AUC and PR-AUC (average precision) via `sklearn.metrics`.
- **E2:** label each scored proposal positive if IoU ≥ `proposal_eval.iou_positive` with any GT box of its frame. Report the natural imbalance.
- **E4 / TrackEval:**
  - Export exactly as `SHARED_CONTRACTS.md` §9: 1-based frames, GT `conf=class=vis=1`, tracker `score,-1,-1,-1`, `seqinfo.ini`, `SKIP_SPLIT_FOL=True`.
  - Run `trackeval.Evaluator` + `trackeval.datasets.MotChallenge2DBox` (`BENCHMARK="MOT17"`-style custom folders, `SPLIT_TO_EVAL` irrelevant with skip) and the metrics `HOTA`, `CLEAR`, `Identity`.
  - HOTA = √(DetA · AssA), averaged over localisation thresholds.
  - ⚠️ Verify the exact TrackEval configuration keys on 1.3.0 with the GT-vs-GT test.
- **Oracle detections:** `gt_oracle` = GT boxes, score 1.0. `gt_noisy` = drop 10 %, Gaussian jitter (σ = 2 px), scores uniform in [0.3, 1.0], fixed seed. Both are clipped to the image.
- **Pooling:** sum counts across matches and recompute ratios. For AP, pool detections across matches. For tracking, report per match and the frame-weighted mean (TrackEval's combined sequences), stating the method.
- **Fair comparison rules:** same frames, GT, IoU thresholds, code, pitch-filter state (reported), and hardware for FPS. A vs B compares systems (YOLO is COCO-pretrained), not equal-data architectures.

## 4. Files you own (edit only these)
- `src/soccertrack/contracts.py`, `metrics.py`, `evaluate.py`, `compare.py`, `explain.py`
- `scripts/run_evaluation.py`
- `configs/evaluation.yaml`
- `tests/conftest.py`, `tests/fixtures/**`, `tests/test_contracts.py`, `test_metrics.py`, `test_evaluate.py`, `test_explain.py`
- `docs/team/SHARED_CONTRACTS.md` (changes need approval from all four members)
- `docs/phase2/sections/member_3_evaluation.md`, `docs/phase3/sections/member_3_evaluation.md`

## 5. Shared files you may read/import but must not edit
- `config.py` (Member 4)
- `data_io.py` (`load_gt_tracks`) and `split.py` (Member 1)
- `classifiers.py`: Phase 3, public attributes only: `predict_proba`, `model`, `gradcam_layer`, `preprocess`, `hog_weights()`
- other configs; notebooks; docs.
- **Never** import `detector_scratch.py`, `detector_yolo.py`, `tracker_iou.py` or `tracker_bytetrack.py`. You evaluate their **files**, not their code.

## 6. Inputs and outputs

| You read | You write |
|---|---|
| Detection / Track / Proposal / Patch score tables + metadata (any producer) | `outputs/metrics/<run_id>.json` (§8) |
| GT via `data_io.load_gt_tracks` (real data) or `tests/fixtures/tracks_gt.csv` | `outputs/detections/gt_oracle|gt_noisy/<split>/<match>__<frameset>.csv` + meta |
| `configs/evaluation.yaml` | `outputs/trackeval/<run_id>/` |
| | `outputs/tables/*.csv`, `outputs/figures/eval/*.png` |
| | `outputs/test_log.md` entries (pre-run config hash) |

## 7. Data access
- You need **no video** for E1–E4: only the MOT txt files (for GT) and the producers' artefacts.
- On the data machine, GT is read through `config` + `data_io` from `F:\Football-Fatigue-Project\data\external\soccertrack-v2\mot\<id>.txt` (read-only).
- On your own machine, the MOT txt files (~6 MB each) are enough. Download only the txt files you need (see `SETUP_AND_DATA.md`), or work purely with fixtures.

## 8. Dependencies and setup
- `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `trackeval==1.3.0` (known to work with NumPy 2.5.4 on the lead machine), `pycocotools==2.0.11`; PyTorch only for Phase 3 Grad-CAM.

## 9. Milestones

| ID | Target | Deliverable | Acceptance |
|---|---|---|---|
| **M0** | Fri 20, 12:00 | `contracts.py` complete; `metrics.iou_matrix` | `test_contracts.py` (all), IoU tests pass; foundation PR merged first |
| **M1** | Fri 20 (evening) | Metrics + E1, E2, E3 + `write_metrics` | Fixture expectations: detection TP 7 / FP 1 / FN 2, P 0.8750, R 0.7778, F1 0.8235, AP@0.5 0.8667; patches TP 3 / FN 1 / TN 3 / FP 1, ROC-AUC 0.9375 |
| **M2** | Sat 21 | `gt_oracle`, `gt_noisy` writers + `oracle` command | Valid Detection tables for val `win60s` and `smoke` |
| **M3** | Sat 21 → Sun 22 | TrackEval export + E4 | Export equals `tests/fixtures/trackeval_example/` byte for byte; GT vs GT gives 1.0; `tracks_pred.csv` gives MOTA 0.6667, IDF1 0.75, IDSW 1 |
| **M4** | Sun 22 → Tue 24 | Pooling, comparison tables, evaluation figures, reference table | Tables for val; figures for the report |
| **M5** | Mon 23 | Frozen test evaluation and log | One logged evaluation per system; per match + pooled |

## 10. Commands (PowerShell, repository root, `.venv` active)
```powershell
pytest tests/test_contracts.py tests/test_metrics.py tests/test_evaluate.py
python scripts/run_evaluation.py fixtures
python scripts/run_evaluation.py oracle    --split val --frameset win60s
python scripts/run_evaluation.py patches   --detector scratch_hog_svm --split val
python scripts/run_evaluation.py proposals --detector scratch_hog_svm --split val --frameset eval10
python scripts/run_evaluation.py detection --detector yolov8n_coco --split val --frameset eval10
python scripts/run_evaluation.py tracking  --pipeline gt_oracle+iou_hungarian --split val --frameset win60s
python scripts/run_evaluation.py tracking  --pipeline yolov8n_coco+bytetrack  --split val --frameset win60s
python scripts/run_evaluation.py compare   --split val
```

## 11. Tests and acceptance
- Every `skip` in your test files becomes a real test; the hand-computed fixture values in `SHARED_CONTRACTS.md` §15 are the acceptance oracle.
- `test_contracts.py` must also **reject** malformed tables (wrong column order, NaN, negative sizes, score > 1, duplicates, frames outside the frameset).
- Any fixture change (e.g. a new edge case) updates `SHARED_CONTRACTS.md` §15 in the same pull request.

## 12. Working independently with fixtures
All your work through M4 runs on `tests/fixtures/` (GT, detections, predicted tracks, proposals, patch scores, metrics, TrackEval targets). Real-data evaluation only needs a producer's CSV + metadata and the GT txt. You can also evaluate the producers' smoke runs as soon as they exist.

## 13. Expected results and deliverables
- Metrics JSON for every evaluated run (val; frozen test).
- Comparison tables:
  1. E3 detection A vs B (per match + pooled);
  2. E1/E2 classifier table;
  3. E4 tracking table with oracle rows (`gt_oracle+iou_hungarian`, `gt_oracle+bytetrack`, `gt_noisy+…`) and detector rows;
  4. reproduced dataset statistics vs the paper;
  5. reference table: Challenge HOTA 0.59–0.60 (approximately comparable on full clips only), GS-HOTA 12.25/10.96 (**not comparable**).
- Figures: confusion matrices, PR and ROC curves, recall-by-height chart.
- Report section: evaluation protocol, metric definitions, results, comparability discussion.

## 14. Integration requirements
- `contracts.py` (M0) is merged before anything else. Its function names and constants are frozen once merged.
- Evaluations read the `operating_threshold` from the producer's metadata; you never choose thresholds for other members.
- Metrics JSON is the single source of every number in the notebook and report (Member 4 reads it).

## 15. Common mistakes and prohibited changes
- Using a different frame set for A and B; evaluating GT frames that the producer did not process.
- Adding 1 to frames anywhere except the TrackEval export; forgetting the `-1` → `1` conversion for GT flags.
- Averaging ratios across matches instead of pooling counts.
- Computing AP after filtering at the operating threshold.
- Reporting numbers in % in JSON, or clipping MOTA at 0 (contract: bounded ratios in [0, 1]; MOTA ≤ 1 and may be negative; FP/frame and FPS ≥ 0 and may exceed 1; IDSW integer).
- Comparing our HOTA with GS-HOTA; presenting earlier Football Fatigue numbers as our results.
- Running test evaluations without the frozen flag and log entry.
- Editing producers' code or configs; changing a contract without approval from all four members.

## 16. Definition of done
- Contracts merged first and stable.
- All fixture expectations reproduced by tests.
- E1–E4 implemented; E4 verified with the GT-vs-GT and fixture tests.
- Oracle detections available.
- Val and frozen-test Metrics JSON complete.
- Comparison tables and figures delivered to Member 4; report section written.

## 17. Phase 3 extensions
- **Failure analysis (train/val only):**
  - classify each error by height bin, image region (far touchline / goal areas / panorama edges), crowding (GT boxes within a radius), FP type (unlabelled person, shadow/line, duplicate), FN type (merged blob, small/far, goalkeeper);
  - list ID-switch events.
  - Output: a ranked list that Members 1 and 2 use to choose improvements.
- **Explainability:**
  - Grad-CAM on `CnnClassifier` (correct vs misclassified patches);
  - SVM weight maps from `hog_weights()`;
  - permutation importance of HOG / colour / shape feature groups (val PR-AUC drop);
  - optional EigenCAM on YOLO.
- **Final evaluation:** full-clip E4; Δ tables Phase 2 → Phase 3; second and last frozen test access.
