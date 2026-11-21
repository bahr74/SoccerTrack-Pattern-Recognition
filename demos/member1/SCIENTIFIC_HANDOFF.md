# Member 1 scientific workflow handoff ? 2026-11-20

**Current corrective status (2026-11-21):** [CORRECTIVE_HANDOFF.md](CORRECTIVE_HANDOFF.md) supersedes earlier GUI/test/readiness statements below. Historical scientific results remain unchanged.


Verified on `feature/scratch-pipeline`, approved skeleton HEAD `cd31fd4`.
No commits, staging, pushes, merges, dependency changes, downloads or held-out
access were performed. Shared configurations/contracts and other members' files
are unchanged. This continues the earlier M1?M8 implementation; prior source is
archived at `outputs/member1_scientific/audit/20261120T043745Z/prior_member1_source.zip`.
Ten inventoried prior artifacts/checkpoints were verified unchanged by SHA-256.

## Start and understand the system

```powershell
.\.venv\Scripts\python.exe demos/member1/app.py --port 8765
```

Open http://127.0.0.1:8765/ ? **Load Real Validation Demo** ? select Detection or
Tracking ? HOG + SVM / Small CNN / rules. **Browse Video** is an actual browser
file input with authenticated upload, not a path-text workaround. Select the
existing synthetic AVI for a correctness-only example. Only the registered real
demo has GT. The model/checkpoint controls appear when relevant. Save-frame exports
a JPEG; video export contains the first 30 frames and track IDs. Analyzer preview
is capped at 600 sequential frames. Long seeks/replay are slow; no cancellation
or real-time throughput guarantee is provided.

The pipeline takes a full-resolution frame, subtracts a sampled median background,
keeps the pitch, cleans the foreground, filters/splits components into player
candidates, crops RGB patches, classifies them, removes overlapping detections
with NMS and links observations using IoU/Hungarian association. Bad foreground
or missing proposals cannot be repaired by better classification. IDs can break
when detections disappear or players cross; there is no motion/appearance model.

**Pipeline Explorer** shows ten actual saved stages with explanations. Those
images are labeled with their saved frame/configuration; they are not the current
analyzer frame. **Model Comparison** separates patch metrics from detector counts.
**Experiments & Results** shows splits, counts, paired changes, figures and downloads.

## Scientific inventory and coverage

Earlier experiments/checkpoints remain in `outputs/member1_demo/`: synthetic
models, a real pilot using 352 train / 176 validation patches, three CNN epochs,
short-background and full-background 20-frame diagnostics. The prior long-background
result (P 0.7897, R 0.4182, F1 0.5468) is historical evidence with different training
and size fitting; it is not silently relabeled as a paired baseline.

Expanded run: `outputs/member1_scientific/runs/expanded_20261120T044321992233Z`.

- Train matches: **118575, 118576, 118577, 118578, 128058**; validation: **117093**.
- Optional 117092 remains disabled. No held-out video/labels were opened.
- Annotation EDA: **35,932 frames, 790,504 boxes**, 22 rows/frame across six matches.
  Zero locally detected missing values, duplicate rows, degenerate/clipped boxes
  or unexpected player counts. These are annotation counts, not full decoded audits.
- Train frame plan: 0,750,1500,2250,3000,3750,4500,5250 per match. Frame 1500 of
  118575 excluded: zero eligible proposal negatives; no negative-ratio fallback.
- **39 accepted train frames, 3,432 patches (858 positive / 2,574 negative)**.
- Validation patch frames: 100,500,900,1300,1700,2100,2500,2900;
  **704 patches (176 positive / 528 negative)**.
- Real negative mix: configured random/partial/proposal = **0.4/0.3/0.3**, rounded
  per frame by the existing sampler, with exact 3:1 total ratio. Exact sample
  identities, boxes and categories are in NPZ; exact selected indices/exclusions
  are in the manifest. NPZ stride 10 means eligible base stride, not actual spacing.
- Height/width model uses all locally cleaned **training** GT. CNN normalization
  and SVM calibration folds use training only. Validation selects SVM C, CNN epoch
  and segmentation settings. Background estimation uses label-free samples across
  each clip: an offline/transductive preprocessing assumption, not causal streaming.
- Four C values searched: 0.01,0.1,1,10; selected **0.01**. CNN **8 epochs**, CUDA,
  **23,873 parameters**; selected epoch 7 by validation AP. Seed 42.
- Completed CNN/SVM training took approximately 9.28/9.13 seconds respectively;
  video/background/sample preparation dominates the approximately five-minute run.
- Source video header metadata is 4096?1080, 25 FPS. Full sequential decoded counts
  of every complete clip and full-match detection evaluation were **NOT RUN**.

Synthetic smoke is explicitly separate: an artificial single clip, disjoint
train/validation frames, one CPU CNN epoch and 0.7 random/0.3 partial negatives.
It proves execution only and does not replace the real experiment pointer.

## Measured results ? local validation only

All patch methods use the same 704 validation patches, score threshold 0.5.
PR-AUC below means average precision on patches; it is **not detection AP**.

| Patch method | Accuracy | Precision | Recall | F1 | Patch AP |
|---|---:|---:|---:|---:|---:|
| Majority DummyClassifier | 0.7500 | 0 | 0 | 0 | 0.2500 |
| HOG + SVM | 0.9418 | 0.8902 | 0.8750 | 0.8825 | 0.9152 |
| Small CNN | 0.8764 | 0.6718 | 0.9886 | 0.8000 | 0.9788 |

CNN has stronger ranking/recall but poorer precision at fixed 0.5; it is not a
universal winner. Validation is reused for selection, so these are development
estimates under sampled class balance, not unbiased final-test estimates.

Paired segmentation comparison holds models, train-fitted size model, background,
GT, frames **0?19**, score 0.5 and IoU 0.5 constant. Configuration selection used
separate development frames **200/250/300** in the same validation match. Candidate
settings: diff threshold 18, opening 1, closing 5, minimum area 40, size tolerance
[0.35,2.0]. Shared YAML is untouched. This is a local runtime experiment.

| Configuration / HOG detector | Proposal recall @0.5 | Precision | Recall | F1 | TP / FP / FN |
|---|---:|---:|---:|---:|---:|
| Starting configuration | 0.5341 | 0.8588 | 0.4977 | 0.6302 | 219 / 36 / 221 |
| Candidate configuration | 0.7386 | 0.9783 | 0.7182 | 0.8283 | 316 / 7 / 124 |

For the selected candidate on the same 20 frames:

| Detector | Precision | Recall | F1 | Measured processing FPS |
|---|---:|---:|---:|---:|
| Untrained rules | 0.7912 | 0.4477 | 0.5718 | 18.77 |
| Expanded HOG + SVM | 0.9783 | 0.7182 | 0.8283 | 11.37 |
| Expanded CNN | 0.7232 | 0.7364 | 0.7297 | 15.19 |

FPS includes foreground/proposals + scoring/NMS, excludes decode, background,
model loading, overlays, encoding and file I/O. These short measurements are
hardware/cache-sensitive diagnostics, not standardized benchmark throughput.
GUI cached frames report no FPS; replay FPS counts every processed frame.

A separate old-versus-new checkpoint rerun uses exactly the **current candidate
conditions** and size model. Earlier pilot HOG F1 **0.7670** ? expanded HOG **0.8283**;
earlier CNN **0.2114** ? expanded CNN **0.7297**. The earlier published 0.5468 is not
the starting number for this paired comparison. See `supplemental_comparison.json`.

Overlap-based HOG miss attribution: starting foreground/localization **205** and
classification/threshold **16**; candidate **115** and **9** respectively. No report
frame falls into the proposal-filter or NMS categories under this diagnostic;
those categories appear in development examples. This is heuristic attribution,
not proof of causal mechanism. Threshold curves and FP/FN figures are saved.

Tracking produces **17 local IDs / 286 observations** over 20 frames. Local greedy
GT matching finds 282 matched observations, **17/22** GT identities seen, zero
predicted-ID changes for the same matched GT and one gap. This short window does
not test hard crossings/occlusions; missing identities remain a limitation. These
counts are **not** official IDSW, HOTA or IDF1. A broader association audit remains.

Official AP/mAP, HOTA, IDF1, E1?E4 certification and held-out final testing:
**NOT RUN**. No complete milestone or benchmark acceptance is inferred from local
metrics.

## Notebook and artifacts

Source notebook: `demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb`.
It uses Member 1's isolated directory because the final shared notebook is Member
4's. All **16 mandatory headings** appear in order, each with Markdown, executable
code, genuine output in the executed copy and interpretation. Default retraining
is OFF. Reused checkpoints are SHA-256 checked. Real and explicit synthetic modes
both executed top-to-bottom with zero cell errors. Source outputs are cleared.

```powershell
.\.venv\Scripts\python.exe demos/member1/build_notebook.py --execute
.\.venv\Scripts\python.exe demos/member1/build_notebook.py --execute --smoke
```

Open the source in a notebook-capable editor with the project `.venv`, or open the
printed execution directory's standalone HTML report. Notebook/JupyterLab servers
are absent; nbclient/nbconvert/ipykernel are installed and were used without installs.
Kernel specs/runtime files are isolated under ignored D-drive outputs. Execution
emits Windows ZMQ selector/TCP warning messages; execution succeeds on loopback.

Verified real executed copy and HTML:
`outputs/member1_scientific/notebook_executions/20261120T050150927182Z/`.
Synthetic execution directory is recorded in
`outputs/member1_scientific/notebook_smoke_complete_console.txt`.
Each successful directory contains `section_audit.json` with all 16 cells executed.
An earlier failed notebook-debug execution is retained as evidence, not a deliverable.

Within `outputs/member1_scientific/runs/expanded_20261120T044321992233Z`:

- `manifest.json`: exact split, frames, counts, paired metrics and limitations.
- `configuration.json`, `size_model.json`, `svm_search.json`, `training_history.json`.
- `models/hog_svm.joblib`, `models/cnn.pt`: new trusted checkpoints + manifest hashes.
- `patches_train.npz`, `patches_val.npz`, `validation_patch_scores_*.csv`.
- `classification_metrics.json`, `detection_threshold_curves.csv`.
- `detection_baseline/`, `detection_candidate/`, `earlier_models_same_conditions/`:
  local score-preserving detections, metrics and error categories.
- `supplemental_comparison.json`, `identity_assignments_local.csv`, `tracking_local.csv`.
- `stages/01_original.png` through `10_tracking.png`, `gt_overlay.png`, `failure_example.png`.
- `figures/eda_boxes.png`, `eda_spatial_tracks.png`, `class_balance.png`,
  `train_patches.png`, `hog_example.png`, `training_curves.png`,
  `confusion_matrices.png`, `patch_pr_curves.png`, `detection_thresholds.png`,
  `trajectories.png`.
- `annotated_validation.mp4`: 20-frame real track preview.
- `reproducibility_audit.json`: environment, all artifact SHA-256/size records,
  preservation checks, sampling/provenance caveats. It is a **post-execution** audit.

GUI verification screenshots and the 30-frame exported video:
`outputs/member1_scientific/browser_verification/1795150768703420300/`.

## M1?M8 and shared integration status

| Milestone | Verified local state | Required acceptance still outstanding |
|---|---|---|
| M1 data/splits | Parsing/cleaning/frame/split tests pass; public APIs preserved | Foundation PR merge not performed; production calls still require shared config/validators |
| M2 EDA | Pure statistics implemented/tested; six-match local EDA recorded | Official ?11 export, shared config/contract integration and full decoded-count audit |
| M3 masks/proposals/rules | Synthetic regression and real saved intermediates; paired local proposal/detection diagnostic | Official eval10 proposal recall @0.3/@0.5, MWB-0 E3 and shared validation; keypoint pixel-schema fallback unresolved |
| M4 patches | Canonical sampler tested; expanded real 3:1 arrays with explicit exclusions | Shared validate_patch_set + official serialized patch/meta integration |
| M5 HOG/SVM | 756-D features, full C search, calibration/checkpoints, patch figures | OpenCV reference parity failure; shared E1/threshold certification |
| M6 CNN | 23,873 parameters, eight GPU epochs, train-only normalization, saved curves/checkpoint | Shared E1 integration/certification |
| M7 detectors | Real paired rules/SVM/CNN/NMS diagnostics | Official eval10/win60s E2/E3 and shared detector/meta outputs |
| M8 tracker | IoU gate/Hungarian/lifecycle tests and short real example | Official oracle/noisy/best-detector win60s E4 and shared Track validation |

Member 3 dependencies: `contracts.validate_tracks`, `validate_detections`,
`validate_proposals`, `validate_patch_set`, `validate_patch_scores`,
`validate_run_meta`, `validate_metrics`; shared table and metadata read/write;
`metrics.iou_matrix`, detection matching/AP, binary metrics; approved evaluator
and oracle/noisy-GT producers. Member 4 dependencies: `config.repo_root`,
`load_config`, `load_data_sources`, `match_paths`, `weights_path`, `frameset_range`,
`artifact_path`, `config_sha256`, `set_seeds`. EDA box-sample kind and aggregate
patch-score metadata still require owner clarification; no new contracts invented.
Production calls keep these gates; the scientific application uses explicitly
local uncertified adapters. No monkeypatching is used for real experiments.

## Verification and unresolved failure

- Full pytest: **175 passed, 1 failed, 44 skipped, 5 deselected**, 14.60 s.
- Member 1 focused: **150 passed, 1 failed, 3 skipped, 2 deselected**, 13.62 s.
- New GUI/science regression group: **10 passed**, including actual upload/decode,
  missing auth, invalid video and traversal/held-out rejections.
- Sole failure: `tests/test_patches_features.py::test_05`, line 56:
  installed OpenCV 5 lacks core `HOGDescriptor`; parity is unverified. Not hidden.
- Three Member 1 skips: shared proposal IoU/recall, official detector/meta validation,
  Track validation. Other **41** selected placeholders belong to other members.
- Five deselections are project data/GPU-marked acceptance checks; the bounded
  real CUDA experiment is separately verified, not a substitute for those checks.
- Logs: `outputs/member1_scientific/pytest_full.txt`, `pytest_focused.txt`,
  `pytest_new_v3.txt`. Initial debug failures and default Windows temp/cache
  permission errors are preserved; the final runs use fresh ignored D-drive paths.
- Genuine headless Edge/CDP browser verification: file-input upload, real loading,
  play/pause, range seek, next/previous, speed, width, all three trained-checkpoint
  sources, SVM/CNN, tracking, JPEG save/download, 30-frame MP4 export, four views
  and all ten loaded stage images. Native OS chooser appearance is not tested.
  Video export was decoded sequentially with OpenCV; all player/browser codecs
  were not tested. `mp4v` support varies.
- Notebook: all 16 sections executed in real and synthetic modes; raw source remains
  clear of outputs. `git diff --check` passed; source syntax/import audit completed.

See `HOG_REFERENCE_ISSUE.md`: OpenCV 5 moved HOG to contrib/xobjdetect; recommend
an owner-approved separate OpenCV 4.12.0.88 / NumPy 2.2.6 reference environment,
not a current-env downgrade or co-installed cv2 provider. No change was executed.
API availability alone cannot certify the NumPy HOG numerical tolerance.

## Complete changed-file / manual staging list

These **35** source paths are modified or untracked relative to HEAD, including
preserved prior implementation work. Review and stage manually only if desired;
**nothing was staged**. The notebook, GUI and new tests are isolated Member 1 work.

- `demos/member1/HOG_REFERENCE_ISSUE.md`
- `demos/member1/README.md`
- `demos/member1/SCIENTIFIC_HANDOFF.md`
- `demos/member1/app.py`
- `demos/member1/build_notebook.py`
- `demos/member1/experiment.py`
- `demos/member1/notebook_support.py`
- `demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb`
- `demos/member1/scientific_workflow.py`
- `demos/member1/supplemental_audit.py`
- `demos/member1/ui_support.py`
- `demos/member1/verify_browser.py`
- `demos/member1/web/app.css`
- `demos/member1/web/app.js`
- `demos/member1/web/index.html`
- `docs/phase2/sections/member_1_scratch.md`
- `scripts/run_scratch.py`
- `src/soccertrack/classifiers.py`
- `src/soccertrack/data_io.py`
- `src/soccertrack/detector_scratch.py`
- `src/soccertrack/features.py`
- `src/soccertrack/patches.py`
- `src/soccertrack/proposals.py`
- `src/soccertrack/segmentation.py`
- `src/soccertrack/split.py`
- `src/soccertrack/tracker_iou.py`
- `tests/test_classifiers.py`
- `tests/test_data_io.py`
- `tests/test_detector_scratch.py`
- `tests/test_member1_demo.py`
- `tests/test_member1_scientific.py`
- `tests/test_patches_features.py`
- `tests/test_segmentation_proposals.py`
- `tests/test_split.py`
- `tests/test_tracker_iou.py`

Do not stage `outputs/`, raw data, patch arrays, model checkpoints, generated
executed notebooks/HTML, browser profiles or the absolute-path local registry.
Manual option after review: use `git add -- <the exact source paths above>`;
no staging command was executed. There is no PR/merge/final-test authorization.
