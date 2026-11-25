# Phase2 report section: Member 3

Owner: Member 3. Only Member 3 edits this file; Member 4 copies the final text into the report, notebook and slides.

Topics: Evaluation protocol E1-E4, metrics definitions, results tables, confusion matrices, comparison with paper and Challenge.

Template for each topic:
1. Method (what and why, with equations where relevant)
2. Settings (from configs/, chosen on validation)
3. Results (reference Metrics JSON run_ids and figure paths; never type numbers by hand)
4. Limitations

---

## 0. Implementation status (Phase 2)

| Milestone | Content | Status | Evidence |
|---|---|---|---|
| M0 | `contracts.py` (validators, table/metadata I/O, identifiers), `metrics.iou_matrix` | Done | `tests/test_contracts.py`, `tests/test_metrics.py::test_01` |
| M1 | Metrics (matching, P/R/F1, AP, classification metrics), E1, E2, E3, `write_metrics` | Done | `tests/test_metrics.py`, `tests/test_evaluate.py::test_01`, `test_02` |
| M2 | Oracle inputs `gt_oracle`, `gt_noisy` and their writer | Done (fixtures); real val run pending on the lead machine | `tests/test_evaluate.py::test_03` |
| M3 | TrackEval export and E4 | Done | `tests/test_evaluate.py::test_04`–`test_06` |
| M4 | Pooling, comparison tables, evaluation figures, reference table, runner commands | Done (fixtures); real tables pending the producers' runs | `tests/test_evaluate.py::test_07`, `test_compare_tables_and_figures` |
| M5 | Frozen test evaluation and log | Pending (lead machine, after the configuration freeze) | — |

All Member 3 tests run on the committed synthetic fixtures (`tests/fixtures/`, Contracts §15), with no dataset, model or GPU: `pytest tests/test_contracts.py tests/test_metrics.py tests/test_evaluate.py`.
`python scripts/run_evaluation.py fixtures` runs every evaluation end to end on the fixtures and writes example tables and figures to `outputs/fixtures_check/`.

**Dependencies on other members.** Commands on real data need `config.load_data_sources`, `config.match_paths` and `config.config_sha256` (Member 4, M0) and `data_io.load_gt_tracks` (Member 1). Until they are merged, the runner stops with a `BLOCKED:` message, and the evaluation reads `configs/evaluation.yaml` and `configs/project.yaml` directly. Metrics JSON written before Member 4 M0 carries a placeholder `config_sha256` and a `provenance.note` saying so.

---

## 1. Evaluation protocol and fairness

### Method
Both pipelines are evaluated by the **same code** on the **same frames**, the **same ground truth** and the **same thresholds**. The evaluator reads only the producers' output files (Detection, Track, Proposal and Patch score tables with their `.meta.json`); it never imports detector or tracker code. Every file is checked against `docs/team/SHARED_CONTRACTS.md` by `contracts.py` before it is evaluated. A file that fails is fixed by its producer; the validator is never loosened.

Fairness rules applied in the code:
- **Same frames.** GT is restricted to the frameset declared in the producer's metadata (`eval10` = every 10th frame for detection; `win60s` = first 60 s for tracking).
- **Thresholds are chosen on validation by the method owner** and read from the metadata (`operating_threshold`); the evaluator never picks a threshold for another member.
- **AP uses all detections** (score ≥ `score_floor` = 0.05), never only those above the operating threshold.
- **Pooling sums counts** across matches and recomputes the ratios; ratios are never averaged across matches.
- **Match-level split** (train / val 117093 / test 128057 + 132831): no frame of an evaluation match is ever used for fitting or tuning.

### Settings
`configs/evaluation.yaml`: framesets, `score_floor` 0.05, IoU thresholds [0.5, 0.3], greedy matching by score, all-point AP, AP IoU range 0.50:0.05:0.95, height bins (0, 30], (30, 45], (45, 60], (60, ∞) px, `proposal_eval.iou_positive` 0.5, noisy-oracle drop 0.10 / jitter 2 px / seed 42, TrackEval metrics HOTA, CLEAR and Identity with similarity threshold 0.5.

### Limitations
- A vs B compares **systems**, not architectures trained on equal data: YOLOv8n is COCO-pretrained, Approach A is trained only on our train matches.
- Phase 2 tracking uses the first 60 s (`win60s`), not full 4-minute clips.

---

## 2. Metric definitions

**IoU.** For boxes A and B (x, y, w, h in full-frame pixels):

  IoU(A, B) = area(A ∩ B) / area(A ∪ B) = I / (area(A) + area(B) − I)

**Matching (one frame).** Detections are sorted by score, highest first. Each takes the still-unmatched GT box with the highest IoU if IoU ≥ threshold (true positive, TP); otherwise it is a false positive (FP). Each GT box is matched at most once, so duplicate boxes on one player are FPs. Unmatched GT boxes are false negatives (FN).

**Precision, recall, F1.**

  P = TP / (TP + FP),  R = TP / (TP + FN),  F1 = 2PR / (P + R)  (0/0 := 0)

**FP per frame** = FP / number of evaluated frames.

**Average precision (AP).** All detections of all evaluated frames are ranked by score. Precision and recall are computed after each detection; precision is replaced by its envelope p̂(r) = max_{r' ≥ r} p(r'); AP is the area under this step curve (all-point interpolation):

  AP = Σ_k (r_k − r_{k−1}) · p̂(r_k)

AP@[.5:.95] is the mean AP over IoU thresholds 0.50, 0.55, …, 0.95. Our AP agrees with `pycocotools` (101-point interpolation) within 0.01 (`tests/test_metrics.py::test_05`).

**Binary classification (E1, E2).** score ≥ threshold → "player".

  Sensitivity = TP / (TP + FN),  Specificity = TN / (TN + FP),  Accuracy = (TP + TN) / N

ROC-AUC (probability that a random player patch scores higher than a random background patch) and PR-AUC (average precision of the scores) come from `sklearn.metrics`. PR-AUC is the more informative one when background dominates (E2).

**Tracking (E4, TrackEval 1.3.0).**
- MOTA = 1 − (FN + FP + IDSW) / GT  (≤ 1, can be negative)
- IDF1 = 2·IDTP / (2·IDTP + IDFP + IDFN)  (F1 at identity level over the whole sequence)
- IDSW = number of times the predicted ID assigned to a GT player changes
- HOTA_α = √(DetA_α · AssA_α), averaged over localisation thresholds α = 0.05, 0.10, …, 0.95. DetA measures detection, AssA association (do IDs stay on the same player), LocA box localisation.

**Value domains** (stored in Metrics JSON): bounded ratios as fractions in [0, 1]; MOTA ≤ 1 and possibly negative; FP/frame and FPS ≥ 0; IDSW and all counts non-negative integers; undefined values are `null`, never 0.

---

## 3. The four evaluations

| ID | Question | Input (producer) | Ground truth / label | Metrics | run_id pattern |
|---|---|---|---|---|---|
| E1 | How good is the patch classifier on balanced patches? | Patch score table (Member 1) | `y_true` of each patch | Confusion matrix, sensitivity, specificity, precision, F1, accuracy, ROC-AUC, PR-AUC; accuracy per patch kind | `E1__<detector>__<split>__pooled__patches` |
| E2 | How does it behave on the real, imbalanced candidates? | Scored proposals before NMS (Member 1) | Positive if IoU ≥ 0.5 with a GT box of the same frame | Same as E1, plus counts of proposals, positives and GT | `E2__<detector>__<split>__<match>__eval10` |
| E3 | How many players are found, and where? | Detection tables (Members 1, 2; oracle: Member 3) | GT boxes on the `eval10` frames | P/R/F1 at IoU 0.5 (and 0.3), AP@0.5, AP@[.5:.95], recall by box height, FP/frame, FPS | `E3__<detector>__<split>__<match or pooled>__eval10` |
| E4 | Do the IDs stay with the right player? | Track tables (Members 1, 2) | GT tracks on the `win60s` frames | HOTA, DetA, AssA, LocA, IDF1, MOTA, IDSW | `E4__<detector>+<tracker>__<split>__<match or pooled>__win60s` |

**Oracle inputs for E4.** To separate tracking errors from detection errors, both trackers are also run on:
- `gt_oracle`: the GT boxes without IDs, score 1.0 (a perfect detector);
- `gt_noisy`: GT with 10 % of boxes dropped, Gaussian jitter σ = 2 px on x, y, w, h, scores uniform in [0.3, 1.0], seed 42, clipped to the image.

With `gt_oracle` every remaining error is an association error of the tracker itself.

**TrackEval export.** Our tables are 0-based; TrackEval (MOTChallenge format) is 1-based. `evaluate.export_trackeval` is the only place where frames become 1-based (`frame_te = frame − start + 1`). GT lines end with `1,1,1` (consider, pedestrian, visible), tracker lines with `score,-1,-1,-1`. Run with `SKIP_SPLIT_FOL=True` and a per-sequence `seqinfo.ini`.

---

## 4. Verification of the evaluation code

The evaluation code is checked against hand-computed answers on the synthetic fixtures (3 frames, 3 players, 640 × 360). These are tests of the code, not experimental results.

| Check | Expected (Contracts §15) | Test |
|---|---|---|
| IoU of (500,140,18,40) and (510,150,18,40) | 0.2 | `test_metrics.py::test_01` |
| Detections at IoU 0.5, threshold 0.5 | TP 7, FP 1, FN 2; P 0.8750, R 0.7778, F1 0.8235 | `test_metrics.py::test_02`, `test_evaluate.py::test_01` |
| All-point AP@0.5 | 0.8667 | `test_metrics.py::test_03` |
| Patch scores at threshold 0.5 | TP 3, FN 1, TN 3, FP 1; ROC-AUC 0.9375 | `test_metrics.py::test_04` |
| Own AP vs `pycocotools` | difference ≤ 0.01 | `test_metrics.py::test_05` |
| GT as detections / GT as tracks | AP = 1; HOTA = IDF1 = MOTA = 1 | `test_evaluate.py::test_gt_as_prediction_is_perfect`, `test_05` |
| TrackEval export | byte-identical to `tests/fixtures/trackeval_example/` | `test_evaluate.py::test_04` |
| Predicted tracks with one ID switch | MOTA 0.6667, IDF1 0.7500, IDSW 1 | `test_evaluate.py::test_06` |
| Pooling | P and R recomputed from summed counts | `test_evaluate.py::test_07` |

---

## 5. Results (validation, match 117093)

Produced on the lead machine with `scripts/run_evaluation.py`. Every number in the report is read from the Metrics JSON in `outputs/metrics/`; nothing is typed by hand.

| Table / figure | File | Built from |
|---|---|---|
| Detection A vs B (per match + pooled) | `outputs/tables/detection_val.csv` | `E3__*__val__*__eval10.json` |
| Classifiers (E1, E2) | `outputs/tables/classifiers_val.csv` | `E1__*`, `E2__*` |
| Tracking incl. oracle rows | `outputs/tables/tracking_val.csv` | `E4__*__val__*__win60s.json` |
| Dataset statistics vs paper | `outputs/tables/dataset_statistics_val.csv` | `outputs/eda/stats.json` (Member 1) |
| References with comparability labels | `outputs/tables/references.csv` | `compare.reference_table()` |
| Confusion matrices | `outputs/figures/eval/confusion_<run_id>.png` | E1, E2 |
| PR curves (detection) | `outputs/figures/eval/pr_detection_val.png` | E3 |
| Recall by box height | `outputs/figures/eval/recall_by_height_val.png` | E3 |
| ROC / PR curves (classifiers) | `outputs/figures/eval/roc_classifiers_val.png`, `pr_classifiers_val.png` | E1, E2 |

Commands (lead machine, after the producers' runs are merged):
```powershell
python scripts/run_evaluation.py oracle    --split val --frameset win60s
python scripts/run_evaluation.py patches   --detector scratch_hog_svm --split val
python scripts/run_evaluation.py proposals --detector scratch_hog_svm --split val --frameset eval10
python scripts/run_evaluation.py detection --detector scratch_hog_svm --split val --frameset eval10
python scripts/run_evaluation.py detection --detector yolov8n_coco    --split val --frameset eval10
python scripts/run_evaluation.py tracking  --pipeline gt_oracle+iou_hungarian    --split val --frameset win60s
python scripts/run_evaluation.py tracking  --pipeline gt_oracle+bytetrack        --split val --frameset win60s
python scripts/run_evaluation.py tracking  --pipeline scratch_hog_svm+iou_hungarian --split val --frameset win60s
python scripts/run_evaluation.py tracking  --pipeline yolov8n_coco+bytetrack     --split val --frameset win60s
python scripts/run_evaluation.py compare   --split val
```

TODO (after the runs): one paragraph per table interpreting the numbers in `outputs/tables/*_val.csv`.

---

## 6. Comparison with the paper and the SoccerTrack Challenge

| Reference | Value | Comparable? | Why |
|---|---|---|---|
| Paper MOT subset statistics | 59,842 frames; 1,316,524 boxes; median box 25 × 55 px | Yes | Reproducible from the MOT files; shown next to our EDA statistics |
| SoccerTrack Challenge 2025 top-5 HOTA | 0.59–0.60 | Approximately | Same clips and test matches, but full 4-minute clips; exact configuration not fully documented. Upper bound for tracking only |
| GTATrack ByteTrack baseline | HOTA 0.42 | Indicative only | Their own validation split |
| Paper GSR baseline GS-HOTA | 12.25 / 10.96 | **No** | Different task: pitch-space, attribute-aware, 45-minute halves |
| Paper BAS mAP@1s | 0.222 / 0.599 | **No** | Different task (ball action spotting) |

Our HOTA (image-space player tracking on 60-s windows) is never compared with GS-HOTA.

---

## 7. Limitations

- **Validation only in Phase 2.** The frozen test evaluation (M5) runs once, on the lead machine, with `--frozen-test` and a logged configuration hash.
- **Short tracking windows.** `win60s` underestimates long-term ID switches; full clips are evaluated in Phase 3.
- **Single validation match.** Results on 117093 may not represent other camera geometries (e.g. 132831).
- **Pooled AUCs.** Pooled E1/E2 ROC-AUC and PR-AUC are `null` because they need the raw scores of all matches; E1 is already computed on the whole split.
- **HOTA pooling** across matches is the frame-weighted mean of per-match values, while MOTA and IDF1 are recomputed from summed counts (stated in `params.pooling` of each pooled JSON).
