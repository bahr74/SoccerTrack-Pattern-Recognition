# Member 1 corrective handoff - 2026-11-21

Current branch: `feature/scratch-pipeline`. This pass fixes verified local defects;
it does not certify M1-M8 integration or perform M9/frozen testing. No packages,
datasets or weights were downloaded/installed; no shared files/configs/dependency
pins were changed. No Git staging, commits, pushes, merges or PRs were performed.
Existing scientific data and experimental checkpoints were not retrained or replaced.
The required unit suite includes its existing tiny synthetic fitting/checkpoint tests.

## Audit finding disposition

| Finding | Status | Evidence / remaining boundary |
|---|---|---|
| H1 cached starting pitch mask | FIXED | Complete inference profile resolved first; preprocessing/cache keys include configuration, video/background, size model and checkpoint identities. Audited frames and export match saved candidate data. |
| H2 HOG reference parity | BLOCKED | No compatible reference in checked local interpreters. The original test still fails, unchanged. Proposed isolated pins/commands are in HOG_REFERENCE_ISSUE.md; none executed. Numerical parity is not inferred from API availability. |
| M1 duplicate proposal negatives | FIXED disclosure; limitation RETAINED | Notebook/dashboard/report record total/distinct counts and weighting. Historical replacement sampling, arrays, metrics and checkpoints unchanged. |
| M2 oversized upload transport | FIXED for supported browser workflow | Size/extension validation before HTTP, JSON preflight, HTTP 413, useful errors and bounded draining of already-in-flight bytes. Browser sparse oversized fixture sends no upload; 4 MiB rejected-body regression receives the structured error. Arbitrary unbounded noncompliant senders are not drained. |
| M3 resolved settings / metadata | FIXED future recording and replay; historical CLARIFIED | New saved plan records eligibility separately from selected frames, derived seeds, exact background indices/count, resolved settings hash and input/artifact hashes. Historical clarification is a separate supplement. Preparation-only replay ignores current YAML and never trains. Full training reproduction NOT RUN. |
| M4 avoidable replay | FIXED locally | Original/GT skip inference and association; Detection skips association. Tracking uses bounded caches/checkpoints. Sequential grab/retrieve indexing retained. |
| M5 percentile fixture discrepancy | PARTIAL / Member 3 BLOCKED | Fixture 10 versus linear-interpolated 14 remains explicit in a new regression; neither shared fixture nor convention was changed. No official fixture acceptance claimed. |
| L1 browser assertions | FIXED | Pause stability, exact requested seek, configuration/checkpoint provenance, cancellation and exported-vs-displayed predictions asserted. |
| L2 contact sheet aspect ratio | FIXED prospectively and in supplemental presentation | Future sheets retain native aspect. A new corrected frame-zero sheet is supplied; historical PNG remains unchanged. |

## Code changes

`preview.py` is the extracted, bounded preview engine used by the existing app,
not a second application. JSON/configuration and model files are cached by file
identity. Masks/intermediate and prediction keys include resolved configuration,
background pixels, video fingerprint, size model and checkpoint SHA-256. Inference
settings explicitly record selected classifier, CPU device, nine requested
background samples and the illustrative score/tracker threshold 0.5. Model
checkpoint hashes are checked against the expanded manifest.

Raw frames are retrieved by sequential grabs; no timestamp seeking. Original and
GT need no classifier calls, Detection analyzes only the requested frame and
Tracking advances consecutive association inputs. Up to two raw images (retained
original budget 64 MiB), two intermediate tuples, four masks, 256 detection/track
entries and eight association snapshots bound caches. One decoded image must fit
in memory; codec/runtime buffers and per-frame proposal complexity are additional.
Decode failures and cancellation release the preview capture. Reopening after a
work-scale mismatch is required instead of silently rescaling coordinates wrongly.

The existing HTTP app uses that engine for analyzer, saved frames and video export.
Exports resolve the selected profile themselves and save provenance/configuration
JSON plus score-preserving detection and track CSVs. Saved frames have JSON sidecars.
UI states are Processing, Ready, Error and Cancelled. Pending/failed/cancelled
predictions are cleared. Cancellation is cooperative between frames, not an
instant interruption of an OpenCV/Torch call. UTC operational logs are under
`outputs/member1_scientific/corrections/runtime/`, with a per-process size cap.

`provenance.py` handles local resolved plans and duplication summaries;
`scientific_workflow.py` records actual future execution values. Replay loads saved
resolved settings rather than live YAML, validates plan hash/splits/indices/seeds,
and checks annotation/background consistency. `--replay-settings <path>` alone
validates only. `--prepare-only --replay-settings <path>` creates new patch outputs
without training or changing the latest completed-run pointer. Full training replay
exists for separately authorized future use and was NOT executed in this pass.
No deterministic retraining claim is made. Full source-video hashes are not supplied;
video identity uses path/size/mtime plus actual background identity.

## Prediction parity and GUI verification

Focused corrective/API/scientific regressions: **26 passed**, including six explicit
approved-real-data cases. Parameterized regression frames: **7,16,17,18,19**.
Counts, full-frame coordinates, scores, thresholded tracker inputs and track rows
agree with the saved candidate experiment at tolerance 1e-6. The API export test
compares all first-20-frame detection and track tables and verifies the MP4 decodes
all 30 exported frames. No Ground Truth is substituted for classifier predictions.

Fresh genuine Edge/CDP results and screenshots:
`outputs/member1_scientific/browser_verification/1795225259983949300/`.
The browser test checks a real selected file upload, oversized-file rejection before
HTTP, pause stopping advancement, exact fixture seek to frame 5, real tracking seek
to audited frame 16, model/preset configuration changes, source/checkpoint labels,
Cancelled state with no pending image, JPEG saving, MP4 decoding, matching pipeline
identity and frame-16 exported predictions, all views and ten stage images.
Native OS chooser appearance is not tested; `DOM.setFileInputFiles` selects the
actual files through the browser. MP4 codec support is not universal across players.

## Before/after measured latency

Median request seconds, three repetitions, approved 117093, same cached nine-sample
background, CPU/two Torch threads, width640, preloaded weights; setup is recorded
separately in `performance_comparison.json`. After uses the corrected candidate
mask. These are GUI latency measurements, not official inference FPS.

| Request | Before seconds | After seconds |
|---|---:|---:|
| original | 1.705 | 0.046 |
| hog_svm | 2.812 | 0.132 |
| tracks | 2.562 | 2.184 |
| tracking_rewind_7 | 1.013 | 0.257 |

Original/GT do zero classifier/association work. A fresh Detection seek performs
one inference rather than all preceding inferences. Tracking still requires a
correct consecutive history; cached rewinds avoid redundant association.

## Historical preservation, duplication and presentation

Correction evidence: `outputs/member1_scientific/corrections/fix_20261121_011724825902`.
Before/after hashes cover **212 historical result files**, including original run
manifests/metrics/NPZ/checkpoints/figures and prior executed notebooks. The final
preservation check reports PASS with zero changed files. The pre-correction source is
also archived there. Corrections are new, clearly labeled files, not replacements:

- `historical_audit_supplement.json`: retrospective metadata/duplication clarification.
- `historical_resolved_settings.json`: exact intended replay plan, including all six
  background index lists and per-match seeds. This is retrospective, not a claim
  that the original run originally recorded complete metadata.
- `presentation/stages/07_patches.png`: aspect-correct supplemental sheet.
- `performance_comparison.json`: individual timing samples and conditions.
- `historical_hashes_before.json`, `historical_preservation_after.json`.

Train: **3,432 total / 2,782 distinct / 650 repeated copies**. Proposal negatives:
**780 total / 130 distinct**. Validation: **704 total / 580 distinct / 124 repeated
copies**. Proposal negatives: **160 total / 36 distinct**. Duplicates retain their
sampling weights. No cross-split match or exact-pixel overlap was found in the
audit. Unique-patch metrics are sensitivity analyses with altered class balance,
not replacements for historical weighted metrics or official results. Recommend a
new versioned sampling/weighting experiment on broader validation after approval;
never silently deduplicate the existing experiment.

The source notebook retains all 16 sections and cleared outputs. Updated explanatory
cells include duplicate counts, weighting, sensitivity cautions and retrospective
provenance. Cached execution completed **16/16 sections, zero cell errors**;
no experimental model was retrained. A new executed notebook/HTML/section audit is
available at `outputs/member1_scientific/notebook_executions/20261121T013040684472Z/`.
Old executed presentation artifacts remain unchanged. The GUI selects the new
supplemental sheet while retaining historical experiment figures and metrics.

## Test status and remaining acceptance

- Corrective + GUI/API/science group, including real parity: **26 passed**.
- All Member 1 default selected tests: **160 passed, 1 failed, 3 skipped, 8 deselected**.
- Full default suite: **185 passed, 1 failed, 44 skipped, 11 deselected**.
- Sole failure remains `test_patches_features.py::test_05`, missing OpenCV HOG API;
  test/tolerance are unchanged. Numerical parity remains UNVERIFIED.
- Three selected Member 1 skips still require shared proposal IoU/recall,
  detector/meta validation and Track validation. Other 41 selected skips are
  other-owner placeholders. Eleven full-suite deselections comprise the original
  five data/GPU tests plus six new real-data cases, separately run and passed.
- `git diff --check`, syntax/import, source-output clearing and ownership checks
  are required final checks; results are in the corrective evidence directory.

Log directory: `outputs/member1_scientific/corrections/`. Final suite and parity
logs are `pytest_final_suite.txt` and `pytest_final_parity.txt`; Member 1 log is
`pytest_member1_verified.txt`. Browser and notebook logs are retained there too.

Member 3 still owns shared validators/table/meta I/O, IoU/metrics/evaluators,
IDs and oracle/noisy-GT producers, plus the percentile convention/fixture decision.
Member 4 still owns configuration/path/hash/frameset/seed interfaces. EDA box-sample
registration and aggregate patch-score metadata need owner clarification. Official
E1-E4, full eval10/win60s, AP/mAP/HOTA/IDF1 and held-out testing remain NOT RUN.
M1-M8 local implementation evidence does not establish official milestone acceptance.

## Manual staging list - review only

These **40** paths include preserved prior implementation plus the corrective
changes. Nothing was staged. Review exact files before any separately authorized
Git action; do not stage outputs, data, weights, executed notebooks, logs, browser
profiles, sparse fixtures, source registries or temporary caches.

- `demos/member1/CORRECTIVE_HANDOFF.md`
- `demos/member1/HOG_REFERENCE_ISSUE.md`
- `demos/member1/README.md`
- `demos/member1/SCIENTIFIC_HANDOFF.md`
- `demos/member1/app.py`
- `demos/member1/build_notebook.py`
- `demos/member1/corrective_evidence.py`
- `demos/member1/experiment.py`
- `demos/member1/notebook_support.py`
- `demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb`
- `demos/member1/preview.py`
- `demos/member1/provenance.py`
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
- `tests/test_member1_corrective.py`
- `tests/test_member1_demo.py`
- `tests/test_member1_scientific.py`
- `tests/test_patches_features.py`
- `tests/test_segmentation_proposals.py`
- `tests/test_split.py`
- `tests/test_tracker_iou.py`

Readiness: **READY FOR DRAFT PR for code review with explicit known blockers**.
This is not merge readiness or full scientific acceptance. HOG parity, the shared
percentile decision and M0 integration remain unresolved and must be visible in
any eventual draft description. No PR was created.
