"""Generate evidence-backed report and handoff from saved measurements."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import re
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import pandas as pd


def table(df,columns=None):
    if columns: df=df[columns]
    def fmt(value): return f'{value:.4f}' if isinstance(value,float) else str(value)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))


def read_log(path):
    """PowerShell redirects may use UTF-16; do not silently corrupt evidence."""
    raw=Path(path).read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')


def write(run):
    run=Path(run).resolve()
    m=json.loads((run/'manifest.json').read_text())
    rel=run.relative_to(ROOT).as_posix()
    grid=pd.read_csv(run/'reference_threshold_comparison.csv')
    results=pd.DataFrame(m['required_detection_results'])
    tracking=pd.DataFrame(m['tracking_diagnostics'])
    probe=pd.read_csv(run/'warmed_runtime_probe.csv').groupby('configuration').processing_seconds.agg(['mean','median']).reset_index()
    probe['fps']=1/probe['mean']
    tests={name:read_log(ROOT/f'outputs/member2_scientific/{name}').strip().splitlines()[-1]
           for name in ['baseline_pytest.txt','member2_unit_tests.txt','member2_gpu_tests.txt','member2_data_tests.txt','member2_all_tests.txt','full_pytest_final.txt']}
    matrix='''| Requirement | Status | Evidence / boundary |
| --- | --- | --- |
| M1 tiling, coordinates, global NMS | COMPLETE | Synthetic full pixel coverage, alternate geometry, clipping, border duplicates, invalid/empty boxes |
| M2 native-resolution real person inference | IMPLEMENTED — LOCAL VALIDATION ONLY | Verified local weights; frame 0 has 69 detections; GPU acceptance passed; independent overlay inspected; shared Member 4 review pending |
| M3 full eval10 and win60s detection | IMPLEMENTED — LOCAL VALIDATION ONLY | Exact 598/1500 coverage, low score cache, runtime and configuration; official validators/writers/E3 blocked |
| M4 ByteTrack adapter and continuous YOLO sequence | IMPLEMENTED — LOCAL VALIDATION ONLY | Installed API, synthetic motion/misses/reset/crossing/index tests; all 1500 frames; official oracle/noisy/E4 blocked |
| M5 bounded detection/threshold/tracker experiments | PARTIAL | Four paired candidates and three tracker settings; selected detector config; tracker defaults retained pending official evaluation; pitch-on comparison blocked |
| M6 held-out testing | NOT RUN | Prohibited; no held-out clip/annotation access |
| Course data preparation and split | COMPLETE within Member 2 scope | Sequential 0-based validation; native geometry; existing match-level split protected; no new training |
| Course imbalance handling | NOT APPLICABLE to new training | Pretrained inference only; person/player mismatch and threshold trade-offs disclosed |
| Course baseline model training | NOT APPLICABLE to Member 2 Phase 2 | Zero-shot pretrained reference; from-scratch training belongs to Member 1 and was not changed |
| Course architecture/training mechanism explanation | COMPLETE | Loaded model 3,157,200 parameters, Conv/C2f/SPPF, strides 8/16/32; installed BCE/CIoU/DFL/TaskAlignedAssigner inspected; no invented losses |
| Course evaluation/reference comparison | PARTIAL | Local PR/F1 and measured runtime; historical reference differences disclosed; official AP/tracking and certified A/B blocked |
| Course specificity/ROC/confusion matrix | NOT APPLICABLE to detector box background TN | TP/FP/FN reported; patch-classification E1/E2 belong elsewhere; no fabricated TN/specificity |
| Documented notebook and reproducibility | COMPLETE locally | 16 scientific sections, 17 executed code cells, cached artifact verification, HTML and execution evidence |
| Real visual evidence | COMPLETE locally | Original/GT/tiles/local/global/pre/post NMS, paired experiments, score curves, TP/FP/FN, trajectories/lengths, decoded 150-frame video |
| Final shared team notebook/report/slides | BLOCKED | Member 4 integration; isolated notebook/report section supplied |
| Official AP/mAP/HOTA/IDF1/MOTA/ID switches | BLOCKED | Member 3 shared production placeholders on current branch; remote branch deliberately not reviewed/merged |
| Fine-tuning, workload and fatigue | PHASE 3 ONLY / outside scope | No execution or physiological claims |'''
    body=f'''# Member 2 Phase 2 scientific report — 2026-11-25

**Status: PARTIAL — SHARED DEPENDENCIES BLOCKED.** This section reports actual local validation, not official E3/E4 certification or final-test performance. Run `{m['run_id']}`. All generated files are under `{rel}` and ignored by Git.

Initial audit: clean `feature/yolo-pipeline` at merged Member1 revision `9c12876`, containing implementation `8cc58ec`. Member2 production files and all11 acceptance tests were placeholders (default focused run:10 skipped,1 deselected). Python3.12.10 and required installed packages were present. Shared config/contracts/evaluators were inspected and confirmed placeholders. Session authorization overrides individual milestone gates; M6 remains prohibited. Initial pytest temp-directory permission errors were resolved using a fresh workspace-local base temp without modifying source or dependencies.

## Method and data flow

Sequential full-resolution BGR frames → native overlapping tiles → COCO YOLOv8n person class → tile-local xyxy → global offset and clipping → torchvision global NMS → full-frame xywh Detection table → low-score-preserving ByteTrack adapter → Track table. Optional binary pitch masking uses bottom-centre membership; missing approved mask is recorded as off. No shared fallback is installed in production. Local CSVs are explicitly provisional and isolated.

Data are SoccerTrack v2 release v1.2, revision `{m['dataset_revision']}`, validation 117093 only: 4096×1080, 25 fps, 5976 header frames. GT has 22 players/frame. The raw parser and geometry cleaner are Member 1's approved implementations. No held-out data, optional matches, training clips, fine-tuning, new dependencies or downloads were used.

Train IDs remain 118575/118576/118577/118578/128058; optional 117092 off. Test IDs 128057/132831 were not opened. Development frame IDs are exactly `{m['development_frame_ids']}`. They are disjoint from required report frames, but the same validation match is development data; this is not unbiased generalization assessment.

## Pretrained architecture and loss explanation

The actual local checkpoint contains **3,157,200 parameters**, with Conv/C2f backbone, SPPF, upsample/concat multiscale aggregation and Detect outputs at strides 8/16/32. The loaded head has 80 COCO classes and reg_max=16; inference requests class 0 only. Anchor-free refers to absence of fixed anchor-box shapes, while grid anchor points remain. The installed `ultralytics/utils/loss.py` implements BCEWithLogits classification, CIoU regression, DFL and TaskAlignedAssigner. These explain the architecture's training mechanism; the team did not train this model and has no new YOLO training losses, epochs or optimizer results.

Source evidence: installed `ultralytics/cfg/models/v8/yolov8.yaml`, loaded checkpoint module inventory in `architecture.json`, installed `ultralytics/utils/loss.py`, and `ultralytics/trackers/byte_tracker.py`. The original Phase 2 course statement was inspected in `docs/phase1/Screenshot 2026-11-19 171526.png`. The prompt's stale path `docs/team/MEMBER_2_AI_PROMPT.md` resolves to `docs/team/prompts/MEMBER_2_AI_PROMPT.md`. Session authorization overrides that prompt's per-milestone stop gates; frozen M6 remains prohibited.

## Tiling and settings

A 20-pixel player would become 6.25 pixels wide when the whole 4096-pixel panorama shrinks to 1280. Native tiles preserve detail. Edge-aligned last tiles guarantee coverage and may overlap more than the configured minimum. Overlap helps border context but creates duplicate predictions, removed using global NMS. Distinct players can still be over-suppressed.

The reference candidate uses width 1280, height 1080, overlap 64, imgsz 960, reporting threshold 0.40, floor 0.05. Its historical pitch filter could not be reproduced because `outputs/masks/117093_pitch.png` is absent. Historical P/R/runtime values are not copied into current results. Four candidates compare tile width 960/1280, imgsz 960/1280 and overlap 64/128 on 33 identical frames. Per-tile and global NMS are fixed at 0.5, max_det=300, half=false. Global-NMS tuning was not undertaken; pitch-on tuning is blocked.

At the same reporting threshold 0.40, all values below are **provisional local IoU-0.5 greedy matching diagnostics**:

{table(grid,['configuration','threshold','tp','fp','fn','precision','recall','f1'])}

Selection maximizes local development F1 over thresholds 0.05–0.90 at 0.05 increments, with measured processing time breaking ties. `{m['selected_candidate']}` at threshold **0.50** won (local F1 0.8352), only narrowly above the 64-overlap 1280-input candidate (0.8337). This small development set does not establish a robust universal optimum. Selected owned config: 1280×1080 tiles, overlap 128, imgsz1280, operating threshold0.50. The original starting config and all candidates remain in the manifest. This is not a test configuration freeze.

## Required real validation results

{table(results,['frameset','frames','gt','pred','tp','fp','fn','precision','recall','f1','processing_seconds','fps'])}

Exact coverage is recorded in `required_validation/eval10/coverage.json` and `win60s/coverage.json`; empty prediction frames are included. A union of 1948 unique frames reuses 150 overlapping frames, without silently shortening either frameset. Retained detections include scores down to 0.05; the table above applies reporting threshold0.50 only for local matching.

**Official AP/mAP, HOTA, IDF1, MOTA and ID switches: BLOCKED / NOT RUN.** Local counts are checked against the hand-computed shared fixture (TP7/FP1/FN2 at threshold0.5). They are not certified E3. There is no meaningful detector background-box TN count, so specificity/accuracy/ROC-AUC and E1/E2 patch metrics are not manufactured.

## ByteTrack

The installed version is Ultralytics8.4.182. Input adapter exposes NumPy confidence/class/centre-xywh, len and integer/boolean indexing. Top-left contract coordinates are converted once. Returned rows are `[x1,y1,x2,y2,track_id,score,cls,idx]`, with idx referring to the original full input; raw detection geometry and score are recovered from that index.

A fresh tracker processes every consecutive frame, including empty frames. Sparse eval10 is rejected. Kalman state is XYAH with motion; the first association uses IoU cost with optional score fusion against high-score detections, followed by IoU-only low-score recovery. In the installed code, low score is strictly greater than track_low_thresh and below the high threshold. Low-score second association has fixed cost threshold0.5; unconfirmed association uses0.7. track_buffer is a count of update frames, not seconds. Removed/lost/new/confirmed state behavior and reset are tested. No appearance ReID, interpolation of missing detections or guaranteed crossing identities is claimed. Kalman output is clipped by the official producer using metadata bounds; detection mode preserves the associated detector boxes.

The reporting threshold0.50 is **not** applied before ByteTrack. Baseline high/low/new thresholds are0.25/0.10/0.25, buffer30, match_thresh0.8, fuse_score=true, output_box=detection. Official oracle/noisy producers are missing; their experiments are explicitly blocked. Synthetic fixture motion is not presented as official oracle acceptance.

{table(tracking,['configuration','frames','ids','observations','median_observations_per_id','max_observations_per_id','association_seconds','association_fps'])}

Conservative settings raise high/new thresholds to0.4. Long-buffer settings use buffer60 and match_thresh0.9. Both reuse cached full win60s detections. Fewer predicted IDs do not prove better identity accuracy: unannotated-person tracks, fragmentation, loss and wrong associations can all alter this count. Approved starting tracker settings are retained until official E4 supports selection. In particular, 335 predicted IDs do not mean 335 real players or an official ID-switch count.

## Runtime definitions and measurements

Installed ByteTrack `match_thresh=0.8` is an assignment-cost ceiling, not an IoU requirement of0.8. With score fusion the cost is `1 - IoU * detection_score`; the low-score stage uses `1 - IoU` without score fusion.

Full-run processing includes tile copying, Ultralytics preprocessing/model/local NMS and CPU global merge with CUDA synchronization; model loading, decoding and file/overlay export are separate. Cold-start/warmup is included. Per-stage timings are stored per frame, including zero-detection frames. `predict_seconds` is not claimed to be neural-network forward time alone. Pitch filtering is off. The union decode-included run took {m['union_decode_included_seconds']:.4f}s ({m['union_decode_included_fps']:.4f}frames/s); it includes sequential decode/assembly and initial stage images, excludes CSV writing/later plots. Do not add or average incompatible FPS definitions. Subset-specific end-to-end timings are not claimed because decoding was shared.

A GPU acceptance test overlapped the last portion of the required run, so those timing diagnostics may include contention. To compare configurations more fairly, a separate paired **eight-frame warmed probe** excludes model setup/warmup, decoding, writing and overlays:

{table(probe,['configuration','mean','median','fps'])}

`mean`/`median` are processing seconds/frame, not end-to-end time. This is a small timing probe, not a full real-time throughput claim. Peak GPU allocation in the original candidate records was cumulative process peak; the separate probe resets peak statistics per candidate and supplies the correct per-candidate records. GPU: NVIDIA GeForce GTX1650Ti, approximately4GiB. Package versions and disk-space snapshot are in `environment.json`; pip check reports no broken requirements.

## Error analysis and conceptual comparison

Actual full-resolution TP/FP/FN overlays exist for frames0/500/1000. `failure_0_fp_0.jpg` visibly shows an unannotated touchline person holding a board, supporting the person-versus-player mismatch for that example. `failure_1000_fn_6.jpg` shows a white-kit player near a blue-kit player; crowd overlap/localization are plausible difficulties, not proven causal explanations. Native crops preserve source aspect and record global crop origin. These examples do not estimate the proportion of all FP/FN types.

YOLO jointly regresses boxes and classes using pretrained features, whereas Member1's HOG/SVM or small CNN classifies proposal-dependent patches. ByteTrack adds motion and low-score recovery compared with basic IoU/Hungarian links. There is no certified paired A/B result and no unsupported claim of universal superiority. Long gaps, crossings, crowd overlap, tiny objects, background people, stitch geometry and score calibration remain limitations. Trajectories are image-space observations, not calibrated speed or physiological fatigue.

## Reproducibility and evidence

Manifest: `{rel}/manifest.json`; configs, source hashes, approved weight/input hashes, seed42, exact frame IDs, package versions, timestamps and timing definitions are included. `source_snapshot/` archives exactly the three science-producing source files identified by the run hashes; the Git revision is the merged starting revision plus explicitly hashed uncommitted Member2 implementation. `artifact_hashes.json` inventories cached artifacts. This local selected-JSON hash is distinct from Member4's unavailable normalized four-file config hash; it is not a substitute for shared freeze provenance.

Notebook source: `demos/member2/notebooks/member2_yolo_pipeline_experiments.ipynb` (outputs cleared). Executed copy and standalone embedded-image HTML are under the run root. Execution evidence reports17 code cells,17 Markdown cells and zero errors. Notebook default reuses/verifies caches and never starts inference/training. The annotated video is2048×540 at25fps, exact half-size native aspect, and was decoded back to150/150 frames. Stage images, overlap crops, curves, runtime comparison, trajectories and track-length figures are under `stages/` and `figures/`.

## Tests and remaining skips

'''+'\n'.join(f'- `{name}`: `{value}`' for name,value in tests.items())+f'''

Initial default tmp/cache locations caused permission errors; rerunning with a new ignored workspace-local basetemp removed them without modifying other members' tests. The persistent failure is exactly `tests/test_patches_features.py::test_05` (OpenCV5 has no HOGDescriptor). No Member2 placeholder skip remains. The34 remaining skips belong to shared/other-member integration or future-phase work; complete reasons are in `outputs/member2_scientific/full_pytest_final.txt`. GPU and saved-artifact data tests are explicitly run separately. M6, official oracle/noisy/E3/E4 and Phase3 are not run.

## Completion matrix

The protected-file inventory verifies **7,081 source/artifact files unchanged**; see `outputs/member2_scientific/preservation_after.txt`.

Exact test results (logs under `outputs/member2_scientific/`):

{table(pd.DataFrame(list(tests.items()),columns=['log','recorded_summary']))}

The complete Member2-only command overrides marker exclusions using `-o addopts=` and includes approved real validation GPU/saved-artifact checks. Full-suite and focused commands use fresh workspace-local `--basetemp` and `-o cache_dir`, as listed in the README. Mocked shared boundaries do not certify integration.

{matrix}

## Integration handoff

Member1: consume only existing `iter_frames`, `video_info`, `read_mot_raw`, `clean_mot_geometry`, `matches_for`, `split_of`, `require_test_access`, `log_test_access`; shared `load_gt_tracks` still depends on Members3/4. No scratch detector/classifier/tracker import is introduced.

Member3: implement existing frozen `contracts.validate_detections`, `validate_tracks`, `validate_run_meta`, `read_table`, `write_table`, `read_run_meta`, `write_run_meta`, `pipeline_id`, plus approved oracle/noisy and E3/E4 runners. Certify fixture matching/AP and GT-as-tracker identity sanity before official reporting. No competing production evaluator is introduced by local diagnostics.

Member4: implement existing `config` M0 (`repo_root`, config/data-source loading, path/weight/frameset resolution, artifact paths, normalized config hash, seeds), supply configured local registry and inspect overlays. Integrate supplied report/notebook only after shared validation. The current official detect CLI fails at `config.load_config()` with `TODO(Member4,M0)`; the trace is saved separately.

Public detector ID `yolov8n_coco`, tracker ID `bytetrack`, Detection columns `frame,x,y,w,h,score`, Track columns `frame,track_id,x,y,w,h,score`. Official paths stay `outputs/detections/yolov8n_coco/val/117093__<frameset>.csv` and `outputs/tracks/yolov8n_coco+bytetrack/val/117093__win60s.csv`, published only through shared interfaces when ready. Debug max-frame runs use `outputs/debug/`. Unit mocks test the boundary; they do not certify real shared integration. Preserve source revision/weight/config hashes when re-exporting the isolated caches.

Phase3 opportunities: approved train-only tile fine-tuning, tiny-object/overlap ablations with larger development samples, verified pitch-mask comparison, better score calibration and identity evaluation. None is implemented here.

**Verdict: PARTIAL — SHARED DEPENDENCIES BLOCKED.**
'''
    for word in ['Member','Phase','Python','Ultralytics','threshold','buffer','reports','all','to','of','at','is']:
        body=re.sub(r'\b'+word+r'(?=\d)',word+' ',body)
    (ROOT/'docs/phase2/sections/member_2_yolo.md').write_text(body,encoding='utf-8',newline='\n')
    (ROOT/'demos/member2/SCIENTIFIC_HANDOFF.md').write_text(body,encoding='utf-8',newline='\n')
    (ROOT/'demos/member2/COMPLETION_MATRIX.md').write_text('# Member 2 requirement coverage\n\n'+matrix+'\n',encoding='utf-8',newline='\n')
    print('Wrote Member 2 report, handoff, and completion matrix')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    write(parser.parse_args().run)
