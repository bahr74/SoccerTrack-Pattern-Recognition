# Phase2 report section: Member 1

> Latest verified corrective status is the 2026-11-21 update at the end of this file. Earlier pilot, GUI and test statements are historical; official integration remains incomplete.

Owner: Member 1. Only Member 1 edits this file; Member 4 copies the final text into the report, notebook and slides.

Topics: Preprocessing, split and leakage controls, segmentation and proposals, patch dataset and imbalance, HOG+SVM, Small CNN, Approach A detector, IoU+Hungarian tracker.

Template for each topic:
1. Method (what and why, with equations where relevant)
2. Settings (from configs/, chosen on validation)
3. Results (reference Metrics JSON run_ids and figure paths; never type numbers by hand)
4. Limitations

## M1: data access and leakage-safe splits

Implementation and synthetic verification completed on 19 November 2026 on
`feature/scratch-pipeline`, based on the approved skeleton at `cd31fd4`.
Real-data and shared-M0 integration remain unverified.

The M1 results in this section describe the initial checkpoint before the later
authorized hardening and M2/M3 work. Current status and test counts are recorded
under "Autonomous continuation and final audit" below.

### Method and settings

- `read_mot_raw` preserves all ten raw columns, enforces integer frame/identity
  fields and finite numeric values, and keeps box coordinates in full-frame pixels.
- `mot_to_tracks` removes exact duplicate rows and boxes with width or height at
  most one pixel, clips to `(W, H)`, removes boxes made degenerate by clipping,
  sets GT scores to 1.0, and sorts by frame and track identity. Each degenerate
  row is counted once; clipping and degeneracy counts can overlap. Conflicting
  rows for the same frame/identity are rejected. Missing or emptied frames within
  zero through the maximum raw frame contribute to `frames_not_expected`.
- `load_gt_tracks` resolves paths through Member 4's `config.match_paths` and
  calls Member 3's `contracts.validate_tracks` through the conversion function.
  These dependencies are never silently bypassed in production.
- Video metadata reports container header values separately from the authoritative
  sequential decoded count. Frame iteration uses `grab`/`retrieve`, starts at
  zero, yields full-resolution BGR uint8 images, filters requested indices in clip
  order, and releases captures on completion, failure or explicit generator close.
  Repeated requests yield one image per index; indices beyond EOF yield nothing.
  Decoded counts are not cached internally; callers may cache repeated checks.
- Split lists and optional flags come from `configs/project.yaml`. All declared
  matches, including disabled optional entries, must be disjoint. Optional
  matches enter lookup results only when enabled. Test access requires an explicit
  true frozen-test flag. Logging appends UTC time, configuration hash, model hashes
  and artefact paths under the configured repository outputs directory.

### Synthetic verification

The following are test results, not dataset or model-performance measurements;
no evaluation `run_id` or Metrics JSON is produced by M1.

- Clean fixture: nine raw rows convert exactly to `tracks_gt.csv`; all four
  issue counts are zero with three boxes expected per frame.
- Dirty fixture: eleven cleaned rows; `duplicate_rows=1`, `degenerate_boxes=1`,
  `clipped_boxes=1`, `frames_not_expected=1`. The right-border box becomes
  `(630, 160, 10, 45)` in a 640 by 360 image.
- A lossless FFV1 video generated from synthetic frames has six decoded images,
  640 by 360 pixels and 25 fps. Every returned pixel matches its BGR source.
- Additional assertions cover malformed/empty MOT input, clipping and missing
  frames, conflicting identities, official path/validator calls, failed/closed
  video captures, optional splits, held-out access guards, append-only log
  provenance, and failure before log creation when the shared hash is unavailable.

Commands use the existing virtual environment. The original system temporary
directory and pytest cache were inaccessible in the execution sandbox, so tests
used a unique ignored directory under `outputs/` and disabled the cache plugin:

```powershell
New-Item -ItemType Directory -Path outputs -Force | Out-Null
$m1Temp = Join-Path (Get-Location) ('outputs/pytest-m1-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $m1Temp --tb=short -rs tests/test_data_io.py tests/test_split.py
# 51 passed, 1 skipped, 1 deselected

$fullTemp = Join-Path (Get-Location) ('outputs/pytest-full-' + [guid]::NewGuid().ToString('N'))
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider --basetemp $fullTemp --tb=short -rs
# 76 passed, 64 skipped, 5 deselected
```

All eight default-selected M1 placeholder skips have real passing assertions.
The ninth M1 placeholder, the opt-in train/validation data check, now contains real
assertions and is deselected by the default marker expression. The one remaining
skip in Member 1's M1 test files is `compute_eda_stats` (M2), which is untouched.
Full-suite placeholder skips total 23 for Member 1's later milestones, 10 for
Member 2, 20 for Member 3, and 11 for Member 4. A green full run does not establish
implementation of those skipped acceptance criteria.

### Dependencies and limitations

- Member 4's `load_config`, `load_data_sources`, `match_paths`, `repo_root` and
  `config_sha256` remain placeholders. Tests load committed YAML directly or mock
  these official interfaces with temporary paths and synthetic hash values.
- Member 3's `validate_tracks` and other validation/table I/O functions remain
  placeholders. Conversion tests mock only the validator and assert exact table
  results independently. Production conversion still raises until M0 is merged.
- `metrics.iou_matrix` remains a Member 3 placeholder and is not needed by M1.
- Real-data, GPU and end-to-end integration tests were not run. No local
  `configs/data_sources.yaml` exists; no dataset or weights were downloaded or
  accessed. Shared-M0 integration must be verified after its owners deliver it.
- No dependency versions, shared contracts or other owners' files were changed.
  At that checkpoint, M2 and later work had not started. Commits and pushes remain
  unauthorized and have not been performed.

## Autonomous continuation and final audit

### Stage 1: verified M1 review findings fixed

1. Provenance logging validates every artefact before hashing, creating folders
   or opening the log. It rejects POSIX absolute, Windows rooted, drive-qualified,
   UNC, control-character and repository-escaping paths. Accepted relative paths
   are normalized to POSIX format. Regressions verify that an invalid path cannot
   create a log or alter existing log bytes, even after a valid path in the list.
2. The opt-in train/validation check fails explicitly for a configured enabled
   match with a missing MOT or clip file. It prints each checked match ID and
   decoded count. Synthetic regressions cover both missing-file cases and confirm
   that disabled optional and held-out entries are never resolved.
3. MOT files are initially read as string tokens. Frame and ID fields are checked
   with exact decimal arithmetic before int64 conversion. Tests preserve
   `2**53 + 1` and `2**63 - 1` through parsing and cleaning, and reject fractional,
   overflowing and non-finite integer tokens. Box fields remain float64.
4. The test commands above now create `outputs/` before selecting a base directory.

Stage-1 results: 35 focused regression cases passed; the M1 selection had 86
passed, 1 M2 skip and 1 data deselection. The full suite had 111 passed, 64 skipped
and 5 deselected. `git diff --check` passed before advancing.

### Stage 2: M2 statistics core implemented; production output integration pending

`compute_eda_stats` now returns exactly the Contracts section 11 key structure:
UTC creation time, per-match resolution/FPS, annotated and decoded frame counts,
cleaned box/identity counts, boxes per annotated frame, box-size percentiles,
track lengths, issue counts, totals and the paper reference. Totals use annotated
frame counts. Percentiles use linear interpolation; box statistics include zero
counts for annotated frames emptied by cleaning. Decoded frames beyond the last
raw annotation contribute to `frames_not_expected`. Empty annotations, entirely
invalid annotations and clips with no decoded frames raise explicit errors.

Only enabled approved train/validation IDs can be inspected. A nonempty test
request is rejected before loading config, and a held-out ID mislabeled as
train/validation is rejected before path resolution. Unconfigured matches are
excluded from the local subset; configured missing files fail clearly.

The acceptance test uses both raw MOT fixtures and temporary 3-/4-frame lossless
videos. Parsing, cleaning and video decoding are real; only existing shared M0
interfaces are mocked. It verifies the fixture JSON structure, all summary fields,
strict finite JSON serialization and reloading, and missing/empty/coverage/leakage
edge cases. The synthetic totals are 2 matches, 7 annotated/decoded frames and
20 cleaned boxes. They are test results, not official dataset results.

Fixture clarification: the dirty example JSON lists width p05 as 10; linear
interpolation over the actual eleven cleaned widths gives 14. The test asserts
14 and leaves the shared example file untouched. Paper references are the frozen
values in the team contract, not a new external verification.

Stage-2 full suite: 118 passed, 63 skipped, 5 deselected; `git diff --check` passed.
The statistics core is implemented and synthetically verified. The runner and
official EDA artefacts are not complete: shared config/validation M0 is missing,
and `box_sample.csv` has no registered table kind in `contracts.TABLE_SCHEMAS`
despite the contract requiring table I/O through `contracts.write_table`.
No private output kind or direct CSV-writing exception was introduced. Member 3
must clarify the sanctioned I/O for this documented schema before that output
can be integrated. No real files were inspected to explain match 118578's size.

### Stage 3: independent M3 components implemented; remaining stage blocked

- Evenly spaced sequential background sampling, deterministic pixelwise median
  and working-resolution resizing; incomplete sampling and invalid scales fail.
- HSV pitch segmentation with largest component, hole filling and closing;
  manually configured full-resolution polygons map into working coordinates.
- Blurred Lab/gray foreground extraction within the pitch and median/open/close/
  vertical-dilation cleanup, with image/mask/parameter validation.
- Full-resolution nearest-neighbor pitch-mask PNG output as uint8 {0,255}, using
  the configured repository output root and checking path containment.
- Least-squares `SizeModel` fitting and expected-height prediction; identifiable
  finite training inputs are required. Explicit non-train provenance is rejected.
  With the existing API, callers remain responsible for supplying train-only GT
  when a table has no split provenance attached.

M3 tests: 25 passed, 2 skipped, 1 deselected. They cover synthetic player
components, IoU >= 0.8 after morphology in both color spaces, empty masks,
boundaries, resolution changes, sampling, full-resolution PNG bytes and model
coefficient recovery. An initial exact-width assertion conflicted with the
specified blur/morphology operations; it was corrected to explicit component
count, centers and IoU >= 0.8. No existing acceptance criterion was removed.

The following remain deliberately unimplemented or blocked:

- `propose`: the specification requires splitting at 1.6 times expected width,
  but `SizeModel` and `proposals` config define height only. An approved source
  for expected width/aspect ratio is needed; no hidden fitted field or default
  width convention was added.
- `proposal_recall`: the shared IoU function is still unavailable. Its acceptance
  placeholder remains visible; no duplicate shared IoU implementation was added.
- `rule_scores`: size-model agreement is required, but its input interface accepts
  only a Proposal table and config; the exact table has no expected-size field
  and config has no fitted model. The owner/team must clarify how the fitted
  agreement reaches the verifier before implementation.
- MWB-0 `run_detection`, runner dispatch, stage artefact integration and actual
  recall/E3 reporting depend on the above and Members 3/4 M0. Existing acceptance
  placeholders remain unchanged. No detector integration success is claimed.
- Keypoint polygon fallback requires a verified calibration JSON layout and clip
  pixel space; selecting it raises an explicit explanatory NotImplementedError.
  HSV and configured manual polygons are implemented independently.

### Final verification and readiness

The final focused run used the same `.venv` command/flags above with
`tests/test_data_io.py tests/test_split.py tests/test_segmentation_proposals.py`:
118 passed, 0 failed, 2 skipped, 2 deselected. The two selected skips are M3
proposal generation and proposal recall. The final full suite produced
143 passed, 0 failed, 60 skipped, 5 deselected. Of these selected placeholders,
19 belong to Member 1 (M3 and later), 10 to Member 2, 20 to Member 3 and 11 to
Member 4. Data and GPU tests remain unrun; no held-out files were accessed.

The working diff is limited to these eight Member 1-owned files: `data_io.py`,
`split.py`, `segmentation.py`, `proposals.py`, their three existing test files,
and this completion note. Shared contracts, configs, dependency files and other
members' code are unchanged. Test-generated videos, PNGs, logs and synthetic
statistics stay in ignored `outputs/pytest-*` directories. No downloads,
installs, GPU training, real-data processing, commits, pushes or PRs occurred.

Self-review: all four verified M1 defects have regression coverage; shared M0
failures are still propagated in production. Model hashes remain caller supplied,
and decoded frame counts still require an annotation-count check to detect
premature decoder termination. These disclosed limitations have not been promoted
to integration guarantees.

Readiness: ready for a Draft PR describing hardened M1, the tested M2 statistics
core and partial M3 work, with the blockers above explicit. Not ready to merge as
completed M2/M3 or to run official experiments. Next steps are the output-I/O and
proposal/scoring clarifications, shared M0 delivery, then remaining M3 work and
unmocked synthetic integration before approved real train/validation checks.

## Full autonomous M1-M8 continuation: current verified status (20 November 2026)

This section supersedes prior checkpoint statuses and test totals above. Earlier
work was preserved. The active branch remains `feature/scratch-pipeline`.

### Core milestone status

| Milestone | Status | Verified work and remaining gate |
|---|---|---|
| M1 | Complete core | Parsing, cleanup, decoding, split guards and all four review fixes tested; official real-data gate still needs shared M0 |
| M2 | Partial | Statistics core, official runner dispatch and provisional six-match raw EDA/plots/sample; official stats/table integration blocked by M0 |
| M3 | Partial | Background, masks, morphology, split/filtered full-resolution proposals, size model and provisional rules work; shared recall/export integration and keypoint fallback pending |
| M4 | Partial | Crop/sampling/build/save/load implemented, official validation retained; provisional real/synthetic NPZ datasets exist; shared NPZ validation unavailable |
| M5 | Partial | NumPy 756-D HOG, HSV/glyph utilities, train-fold calibrated SVM, validation C selection, checkpoint round trips and real-subset model; OpenCV reference test fails because installed HOGDescriptor is absent |
| M6 | Complete core | 23,873-parameter CNN, train-only normalization, augmentation, weighted BCE/Adam, validation early stopping, checkpoints, curves and Grad-CAM interface tested; bounded real CUDA training completed |
| M7 | Partial | In-memory scratch detection/NMS works and provisional real predictions/video exist; official export calls stay gated by shared M0 |
| M8 | Partial | IoU/Hungarian assignment, gating, confirmation/age/deletion, empty frames, sequence/runner export paths and provisional ID video; official validation/E4 unavailable |
| GUI | Complete bounded demo | Local browser application with original/GT/masks/proposals/rules/trained predictions/IDs, seek/play/step/width controls and annotated export; preview capped at 600 frames, export at 30 |

No M9 frozen evaluation or Phase 3 work was started.

### Implementations and explicit provisional interfaces

- `SizeModel.width_to_height` is an additive, provisional field fitted from median
  train w/h. A height-only model uses the reciprocal midpoint of configured h/w
  bounds. Wide components are partitioned and filtered; coordinates/area map to
  full resolution. Scoring's named in-memory size-agreement context is explicitly
  provisional and is not serialized as a new shared table column.
- Official `proposal_recall` still calls Member 3 IoU. The independent tracker
  owns its geometry implementation; the isolated demo uses it for explicitly
  provisional diagnostics. No shared function was implemented or copied into its
  owner's module, and no shared module is monkeypatched by production/demo runs.
- HOG uses NumPy gradients, bilinear spatial/orientation votes and L2-Hys, ordered
  by block y/x then cell y/x. Its lack of OpenCV Gaussian block weighting is
  documented. Reference tolerance/order comparison remains unverified.
- `fit_with_validation` is an additive Member 1 SVM method because the frozen
  `fit(X,y)` API has no validation arguments. Calibration always uses train folds.
- The production runner dispatches approved commands through shared interfaces.
  It never falls back to demo YAML readers or direct CSV writers. Aggregate
  patch-score metadata and the EDA box-sample I/O gap raise explicit blockers.
- `demos/member1` is the separately authorized provisional application/experiment
  location. Its outputs say PROVISIONAL and stay outside official output folders.
  Direct YAML/raw-label reads, locally checked GT, schema-shaped NPZ/CSV outputs
  and local diagnostics are not presented as shared-contract integration.

### Data and real measurements

Read-only raw-label inspection covered the five enabled training matches and
validation 117093: 6 matches, 35,932 annotated frames, 790,504 raw boxes. Each
observed frame has 22 boxes and each match has 22 identities; no frame gaps, exact duplicates or raw
width/height degeneracy were found. Trailing columns are -1. Full decoded-count
alignment and clipping validation for all six clips remain unverified.

The smaller 118578 text file does NOT mean fewer annotations: it has 6,000 frames
and 132,000 rows, the same counts as 118575. Its 528,000 coordinate tokens have
no decimal points, compared with decimal formatting in all 528,000 tokens of
118575. Observed file sizes are 4,540,066 versus 6,122,283 bytes. Integer-only
coordinate formatting explains the main size difference; no frames were inferred
missing from file size.

The real training diagnostic used approved train match 118575 only: frames
0/10/20/30, 352 patches (88 positives). Validation used separate match 117093,
frames 0/10, 176 patches (44 positives). Checkpoint histories/configs are saved.
The bounded SVM C search used 0.01/0.1/1.0, selecting 0.1. The experiment omits the
10.0 candidate to bound work; unit tests exercise the full configured grid.
The experimental negative mix is 70% random / 30% partial / 0% proposal with 3:1
balance; this is explicitly different from the production builder's approved
40/30/30 mixture. Neither experiment nor its NPZ files are official patch exports.

Three CUDA CNN epochs were actually run. Validation patch diagnostics:

| Model | PR-AUC | ROC-AUC | Precision at .5 | Recall at .5 | F1 at .5 |
|---|---:|---:|---:|---:|---:|
| HOG/SVM | 0.993742 | 0.997762 | 1.000000 | 0.886364 | 0.939759 |
| Small CNN | 0.795377 | 0.928719 | 0.888889 | 0.181818 | 0.301887 |

These are tiny, provisional patch-subset measurements. They do not establish
full-match detector performance, fatigue prediction or official E1 results.

On validation frames 0-19 (440 GT boxes), the short 20-frame background window
had local proposal recall@.5 = 0.061364. A bounded follow-up used nine background
samples across 5,976 frames, taking 22.106 seconds to prepare. Local proposal
recall@.5 rose to 0.531818. With fixed illustrative score threshold .5 and IoU .5,
the full-background HOG/SVM diagnostic produced TP=184, FP=49, FN=256,
precision=0.789700, recall=0.418182, F1=0.546805. Processing/drawing/encoding ran at
3.417 FPS over 20 frames, excluding background preparation. No official E3/E4 or
AP/HOTA/IDF1 result is claimed. Temporal sampling is small; recall and robustness
remain weak and require validation work.

Synthetic models are separately saved and labeled. Their generated train/val
frames share a scene and are diagnostic only. No held-out test file was accessed.

### GUI and artifacts

Launch instructions and walkthrough: `demos/member1/README.md`.

```powershell
.\.venv\Scripts\python.exe demos/member1/app.py --port 8765
# Open http://127.0.0.1:8765/
```

Use the generated synthetic video first. For real validation choose 117093's
video/MOT paths, background window 5976 and Real subset-trained, then compare GT,
foreground, proposals, HOG/SVM, CNN and IDs. Background preparation is bounded;
seeking decodes sequentially and resets tracking on rewind. Model selection and
background are preserved during export. Cached requests report no FPS.

Tkinter could not find its Tcl runtime, so the GUI uses the standard-library
loopback HTTP server and browser HTML/JS; no dependency was installed. HTTP smoke
checks opening, forward/backward seeking, masks/IDs, token rejection, cache/FPS,
export and download. Headless Edge rendered the actual UI, saved `gui_smoke.png`,
and the screenshot was visually inspected. Browser automation initially failed,
then the isolated headless Edge fallback succeeded.

Verified artifact paths (all below `outputs/member1_demo/`):

- `experiment_provisional.json`, `all_available_raw_eda_provisional.json`,
  `eda_provisional.json`, `box_sample_provisional.csv`,
  `eda_box_distributions_provisional.png`.
- `real_subset_models/hog_svm.joblib`, `real_subset_models/cnn.pt`, and equivalent
  `synthetic_models/` checkpoints; each directory also has config/history/curves.
- `real_subset_patchsets/patches_train.npz`, `patches_val.npz`,
  `scores_train_provisional.csv`, `scores_val_provisional.csv`; equivalent synthetic
  NPZ files are in `synthetic_patchsets/`.
- `real_val_full_background_preview/annotated_provisional.mp4`,
  `detections_provisional.csv`, `tracks_provisional.csv`, `overlay_provisional.png`,
  `foreground_provisional.png`, `background_provisional.png`, `pitch_provisional.png`.
- Earlier short-window and synthetic outputs are retained in `real_val_preview/`
  and `synthetic_preview/`; `real_val_preview/gt_vs_predictions_provisional.png`
  shows GT and the earlier short-window predictions side by side.
- `gui_smoke.png`, `artifact_manifest.json`, `staging_suggestions.txt`.

All three main sample MP4s contain 20 frames and were reopened/decoded successfully.
Saved checkpoint SHA-256 values, SVM reloaded scores, repeated CNN CPU reload
predictions and NPZ contents were checked. GUI exports are separately generated
under its `exports/` subdirectory. No checkpoint was fabricated or pretrained.

### Remaining blockers and readiness

The OpenCV HOG reference test remains visibly FAILED: installed OpenCV 5.0 has
no `HOGDescriptor`. It was not skipped or replaced with a weaker assertion.
Member 4 must supply an approved reference environment; package changes remain
unauthorized. Members 3/4 must deliver configuration/path/hash/frameset and shared
validation/table/metadata/IoU interfaces. Keypoint fallback still needs verified
calibration pixel space. Aggregate patch-score sidecar and box-sample table I/O
need clarification. Official Metrics JSON/TrackEval acceptance remains pending.

The new source is functional prototype work, not accepted production integration.
Simple IoU tracking can switch IDs under ambiguous overlap/occlusion. Per-clip
background selection strongly affects detection recall. Trusted local checkpoints
only are loaded. No official model or metrics artifacts were silently substituted.

Readiness: FIX BEFORE PR acceptance. Reviewable prototype/demo work exists, but
the reference-parity failure and shared integration gates must be resolved before
claiming completed M1-M8 acceptance. No commits, staging, pushes, merges, rebases
or PR creation were performed.


### Final execution audit and manual staging inventory

Final focused Member 1 run: **144 passed, 1 failed, 3 skipped, 2 deselected**.
Final full pytest run: **169 passed, 1 failed, 44 skipped, 5 deselected**.
The sole failure is the unavailable OpenCV HOG reference; it is not concealed.
The three remaining selected Member 1 skips are shared proposal-recall, official
export/metadata, and official tracking validation integration. Other members'
41 selected placeholders remain unchanged. Data/GPU-marked acceptance tests were
not run; bounded CUDA training and provisional video experiments were separately
executed and verified. Logs are `outputs/member1_demo/pytest_focused_final.txt`
and `pytest_full_final.txt`.

Syntax/import checks passed for all nine Member 1 modules and the runner/demo.
`git diff --check` passed. The changed-file whitelist contains exactly 22 Member
1-owned/explicitly authorized files; shared code, contracts, configs, dependency
versions and other members' tests are unchanged. Generated artifacts, source
hashes and file sizes are inventoried in `outputs/member1_demo/artifact_manifest.json`.
The GUI HTTP smoke covers export/download and failed-open state cleanup; cached
frames have null FPS and processing rates count all replayed frames. Real video
and MOT identities are checked before experiment labels are read. A failed video
open clears cached content rather than displaying another video's predictions.

Long seeks still require sequential replay and can be slow. The GUI is a bounded
prototype with no background job cancellation; it is not a production video tool.
Remaining approval/integration blockers are listed above; no official E1-E4
acceptance was inferred from local diagnostics.

Recommended source staging list (manual review only; **nothing was staged**):

- `demos/member1/README.md`
- `demos/member1/app.py`
- `demos/member1/experiment.py`
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
- `tests/test_patches_features.py`
- `tests/test_segmentation_proposals.py`
- `tests/test_split.py`
- `tests/test_tracker_iou.py`


## 2026-11-20 expanded workflow update ? local scientific validation

This update supersedes earlier GUI usage and tiny-pilot status statements, while
preserving their evidence. Full details, exact changed-file/manual staging list,
M1?M8 acceptance gaps and commands are in
[Member 1 scientific handoff](../../../demos/member1/SCIENTIFIC_HANDOFF.md).

### Method and protocol

The existing implementation was reused. MOT geometry cleaning is exposed through
an additive pure helper; `mot_to_tracks` still requires shared validation. Detector
intermediate masks/proposals and pre/post-NMS scores are exposed through reusable
helpers with unchanged `detect_frame` signature. SVM search now computes HOG once
per split and records the full grid; checkpoints remain backwards compatible.
No shared contracts, configurations, other owners' code or dependency pins changed.

Run evidence: `outputs/member1_scientific/runs/expanded_20261120T044321992233Z/manifest.json`, configuration, sample plans, exclusion log,
checkpoint hashes, CSV scores, saved stages and `reproducibility_audit.json`.
Train matches are 118575/118576/118577/118578/128058; validation is 117093; optional
117092 remains off. Exact 3:1 balance and configured 0.4/0.3/0.3 negative mixture
are preserved. One frame (118575:1500) was excluded for insufficient proposal
negatives, without fallback. This leaves **3432 train patches /
704 validation patches**, from 39 / 8 accepted frames.
Local EDA covers 35,932 annotated frames / 790,504 boxes, all 22 rows/frame, with
zero locally observed missing, duplicate, degenerate or clipped annotations.
Full decoded counts of all complete videos were not audited.

Size fitting, SVM calibration and CNN normalization use train data only. SVM
searched C=0.01,0.1,1,10 and selected 0.01. CNN completed
8 CUDA epochs with 23873 parameters,
seed 42, using epoch 7 by validation AP. Validation selects C/epoch/configuration;
these are development estimates. Nine-sample label-free backgrounds span each
clip, so this is offline processing, not a causal streaming evaluation.

### Results

Same validation patches, score 0.5; **patch** AP is not detector AP:

| Method | Precision | Recall | F1 | Patch AP |
|---|---:|---:|---:|---:|
| majority_baseline | 0.0000 | 0.0000 | 0.0000 | 0.2500 |
| hog_svm | 0.8902 | 0.8750 | 0.8825 | 0.9152 |
| cnn | 0.6718 | 0.9886 | 0.8000 | 0.9788 |

The explicit sklearn majority DummyClassifier has accuracy 0.75 despite zero
positive recall; accuracy alone is misleading under the sampled imbalance.

Paired configuration comparison on identical frames 0?19, GT/background/size model,
models and score/IoU thresholds 0.5: proposal recall **0.5341 ? 0.7386**, HOG detector
F1 **0.6302 ? 0.8283** (TP/FP/FN 219/36/221 ? 316/7/124). Candidate settings were
selected using separate development frames 200/250/300 in the same validation
match; no final held-out testing occurred. Candidate runtime parameters are saved
in the manifest; approved YAML is unchanged. The candidate reduced the overlap-based
foreground/localization miss category from 205 to 115 and classification/threshold
misses from 16 to 9. Attribution is heuristic, not causal certainty.

Selected candidate, identical report frames:

| Detector | Precision | Recall | F1 | Processing FPS |
|---|---:|---:|---:|---:|
| rules | 0.7912 | 0.4477 | 0.5718 | 18.77 |
| hog_svm | 0.9783 | 0.7182 | 0.8283 | 11.37 |
| cnn | 0.7232 | 0.7364 | 0.7297 | 15.19 |

FPS includes segmentation/proposals, classification and NMS; excludes decode,
background/model preparation, overlays, encoding and I/O. Short sample timing is
not standardized benchmark throughput. With current conditions held constant,
older versus expanded checkpoint HOG F1 is 0.7670 ? 0.8283; CNN 0.2114 ? 0.7297.
This fair checkpoint rerun is separate from the historical pilot number 0.5468.
See `supplemental_comparison.json` for exact conditions.

Tracking yields 17 IDs / 286 observations. Local greedy identity analysis matches
17 of 22 GT identities, finds zero observed predicted-ID changes and one gap in
this short sample. Missing players, occlusion/crossing behavior and broader
association quality remain unresolved. These are not official IDSW/HOTA/IDF1.

Actual figures: `outputs/member1_scientific/runs/expanded_20261120T044321992233Z/figures/` contains EDA, patch/HOG examples, learning curves,
confusion matrices, patch PR curves, detector threshold curves and trajectories.
`stages/` has ten pipeline stages plus GT and FP/FN overlays;
`annotated_validation.mp4` has the actual 20-frame track example.

### Notebook, application and verification

The application has four views: Video Analyzer, Pipeline Explorer, Model Comparison,
Experiments & Results. Browser upload, one-click approved demo, metadata, playback,
seek/step/speed, detector/model sources, saved-frame and video export were genuinely
exercised through headless Edge/CDP inputs. Screenshots and browser results:
`outputs/member1_scientific/browser_verification/1795150768703420300/`.
The MP4 was sequentially decoded as 30 frames; native OS chooser appearance and
universal browser/player codec support were not tested. Analyzer is bounded at
600 frames; exports at 30. Rewinding tracking replays from frame zero.

Source notebook is isolated from Member 4 at
`demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb`.
All 16 mandatory ordered sections executed in real cached and explicitly synthetic
smoke modes with zero cell errors, actual outputs and interpretation. Training is
OFF by default; source outputs are cleared. Executed real notebook, HTML and
16-section audit: `outputs/member1_scientific/notebook_executions/20261120T050150927182Z/`.
Synthetic final execution: `outputs/member1_scientific/notebook_executions/20261120T050643283639Z/`.

Full pytest: **175 passed, 1 failed, 44 skipped, 5 deselected**. Focused Member 1:
**150 passed, 1 failed, 3 skipped, 2 deselected**. New GUI/science regression group:
10 passed. Logs are in `outputs/member1_scientific/pytest_*.txt`.
The one failure is `test_patches_features.py::test_05`: installed OpenCV 5 lacks
`HOGDescriptor`. It remains visible; reference parity is unverified. The
[reference issue](../../../demos/member1/HOG_REFERENCE_ISSUE.md) explains the upstream
API migration, an exact separate-env recommendation and NumPy compatibility impact.
No environment changes were made. Three Member 1 integration skips need Member 3
shared IoU/recall, detector/meta validators and Track validator; the other 41 are
unchanged other-member placeholders. Data/GPU-marked acceptance remains deselected.

### Handoff and limitations

M1?M8 have local implementations/regression evidence, but **official milestone
acceptance is incomplete**. Members 3/4 config loading, shared validators/table/meta
I/O, approved evaluation and oracle/noisy producers remain required. Official
E1?E4, AP/mAP, HOTA/IDF1, complete eval10/win60s and final held-out testing are
**NOT RUN**. Keypoint/pixel fallback schema, EDA box-sample kind and aggregate
patch-score metadata still need owner clarification. No silent shared fallback
was introduced. The notebook/application are local provisional demonstrations.
Ten prior inventoried artifacts/checkpoints retain their SHA-256 hashes.
No Git writes, downloads, installs or shared-file edits were performed.


## 2026-11-21 scientific audit corrective pass

Current corrective handoff and complete staging list:
[CORRECTIVE_HANDOFF.md](../../../demos/member1/CORRECTIVE_HANDOFF.md).

The analyzer now resolves configuration before preprocessing and keys caches by
configuration/background/video/size/checkpoint identity. It no longer reuses the
starting pitch mask under the candidate preset. Regression checks on frames
7,16,17,18,19 and API export agree with saved candidate counts, coordinates,
scores, tracker inputs and track rows within 1e-6. Current frame/model/preset/
checkpoint/configuration/background provenance is visible and saved with exports.
Original/GT skip inference and association; Detection skips association; tracking
uses bounded caches and replay checkpoints. Decode errors/cancellation release
resources. No timestamp seeking was introduced.

Browser size/format validation and upload preflight prevent oversized transfers;
HTTP 413 and bounded rejected-body handling preserve useful errors. Processing,
Ready, Error and Cancelled states are explicit. Browser tests assert pause stability,
exact seek, preset/configuration changes, cancellation, and exported-vs-displayed
predictions. Fresh evidence is in
`outputs/member1_scientific/browser_verification/1795225259983949300/`.

Three-run median frame-19 request latency changed from 1.705 to 0.046 seconds for
Original, 2.812 to 0.132 for HOG Detection, 2.562 to 2.184 for Tracking; rewind to
frame7 changed from 1.013 to 0.257 seconds. These include preview-request work,
use preloaded weights/cached background and are not official benchmark FPS.
Exact samples and setup timings: `outputs/member1_scientific/corrections/fix_20261121_011724825902/performance_comparison.json`.

Historical sampling is unchanged: train 3432 total / 2782 unique / 650 repeated;
validation 704 / 580 / 124. Proposal negatives contain 780 / 130 distinct train
and 160 / 36 distinct validation patches. Replacement sampling changes weights;
it is not demonstrated train/validation leakage. Unique-patch metrics are sensitivity
analyses, not replacement official results. A future sampling improvement must be
separately versioned and approved, preserving current arrays/weights/results.
Notebook/dashboard descriptions now quantify these facts.

Future local runs record resolved settings, actual background indices/effective
counts, eligible versus selected frames, derived seeds and hashes. Saved-settings
replay ignores live YAML; validation and preparation-only behavior are tested
without training. Historical stride10/background60 discrepancies are explained in
new supplemental files under `outputs/member1_scientific/corrections/fix_20261121_011724825902`; original metadata/results remain
unchanged. A new aspect-correct patch sheet replaces only the presentation choice.

The source notebook is cleared and all16 sections passed cached execution without
retraining. New executed notebook/HTML/section audit:
`outputs/member1_scientific/notebook_executions/20261121T013040684472Z/`.

Corrective regression group: 26 passed (including six real parity/API cases).
Member 1 default: 160 passed, 1 failed, 3 skipped, 8 deselected. Full default:
185 passed, 1 failed, 44 skipped, 11 deselected. The sole failure is the unchanged
OpenCV HOG reference check. Known local interpreters provided no compatible API;
exact proposed isolated pins/commands and required numerical closure evidence are
in [HOG_REFERENCE_ISSUE.md](../../../demos/member1/HOG_REFERENCE_ISSUE.md).
No environment was installed. Percentile fixture10 versus linear14 remains explicit
pending Member3 approval; shared fixtures/conventions were not edited.

Official M0/E1-E4, full eval10/win60s, AP/mAP/HOTA/IDF1 and frozen testing remain
incomplete/NOT RUN. Recommended readiness is a Draft PR for review with these
blockers explicit, not merge or milestone acceptance. No experimental models were
retrained; no shared-file edits, downloads, package changes or Git writes occurred.
