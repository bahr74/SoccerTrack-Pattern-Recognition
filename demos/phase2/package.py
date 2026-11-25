"""Consolidate verified final artifacts and manual acceptance instructions."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,hashlib,json,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from demos.phase2.workflow import read,write,sha,copy_new

def logtext(path):
    raw=Path(path).read_bytes();return raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
def summary(path):
    text=logtext(path);rows=[r for r in text.splitlines() if re.search(r'\d+ passed',r)];return rows[-1] if rows else 'NOT COMPLETED; inspect log'
def git(*args):return subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
def mdtable(rows,columns):
    fmt=lambda v:f'{v:.4f}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(c,'')) for c in columns)+' |' for r in rows)
def package(run):
    run=run.resolve();data=read(run/'accepted_results.json');notebook=read(run/'notebook_verification.json')
    rel=run.relative_to(ROOT).as_posix();source='notebooks/phase2_baseline.ipynb'
    def doc(name,text):(run/name).write_text(text.strip()+'\n',encoding='utf-8')
    e3=[dict(method=d['method'],**d['metrics']) for d in data['results'] if d['eval_id']=='E3']
    e4=[dict(method=d['method'],**d['metrics']) for d in data['results'] if d['eval_id']=='E4']
    e1=[dict(method='Majority negative (auxiliary)',**data['auxiliary_majority_baseline']['metrics'])]+[dict(method=d['method'],**d['metrics']) for d in data['results'] if d['eval_id']=='E1']
    e2=[dict(method=d['method'],proposals=d['counts']['proposals'],recall50=d['breakdown']['proposal_recall_at_iou']['0.5'],recall30=d['breakdown']['proposal_recall_at_iou']['0.3'],f1=d['metrics']['f1']) for d in data['results'] if d['eval_id']=='E2']
    tests=summary(run/'authorized_final_tests.log');real=summary(run/'accepted_real.log')
    default_command=f'.venv/Scripts/python.exe -B -m pytest -q -rs --deselect tests/test_config.py::test_07_download_dry_run --basetemp={rel}/manual_test_tmp -o cache_dir={rel}/pytest_cache --tb=short'
    doc('START_HERE.md',f'''# SoccerTrack Phase 2 manual acceptance

Repository: D:\\pattern_prj. Run commands from this directory. The accepted package is `{run}`. No installation, training, downloads or held-out access is needed to demonstrate it.

## Launch the single application

```powershell
Set-Location -LiteralPath 'D:\\pattern_prj'
.venv/Scripts/python.exe demos/phase2/launch.py --run {rel} --port 8765
```

Open http://127.0.0.1:8765. If the server is already running from this task, open the URL directly. If port8765 belongs to another application, choose `--port 8766` and open that exact port; do not terminate an unrelated process. Ctrl+C stops this local server.

The Integrated validation tab loads accepted E1–E4 and cache-backed validation views. Original/GT skip inference; tracking views use selected CNN/IoU or conservative YOLO/ByteTrack. Frames0–1499 have full cached detections; beyond1499 select stride10 frames. Tracking covers0–1499 only. Legacy tabs retain live scratch uploads/exploration and are explicitly historical/live, not the official result tables. Uploaded videos without annotations disable GT; YOLO/ByteTrack cache views refer only to validation117093. Image click enables zoom. Interactive filters do not rewrite official results.

## Open the scientific deliverables

```powershell
Start-Process '{run / 'phase2_baseline_executed.html'}'
Start-Process '{run / 'phase2_baseline_executed.ipynb'}'
Start-Process '{run / 'report/phase2_report.html'}'
Start-Process '{run / 'report/phase2_report.md'}'
Start-Process '{run / 'videos/synchronized_comparison.mp4'}'
Start-Process '{run / 'tables'}'
Start-Process '{run / 'figures'}'
```

Canonical editable notebook: `{ROOT / source}`. Executed notebook: `{run / 'phase2_baseline_executed.ipynb'}`. HTML embeds its images. Report formats available: HTML and editable Markdown; PDF/DOCX are NOT generated. To obtain PDF manually, open the HTML in an installed browser or Word and print/export to PDF, then review page breaks. The attempted renderer was blocked by unsigned PowerShell script policy, and no policy/dependency change was made.

## Tests

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR='{run / 'mpl'}'
$env:YOLO_CONFIG_DIR='{run / 'yolo'}'
{default_command}
```

Final authorized result: {tests}. Bounded real/GPU/artifact checks: {real}. The sole expected failure is the missing OpenCV HOG reference API; five skips are Phase3. One inherited downloader dry-run test is explicitly deselected because this task forbids executing that script.

## Reproduce only when necessary

The completed `stage.json` prevents accidental repetition. Existing model weights are immutable. For an explicitly intended new run, choose a fresh `outputs/phase2_final/<version>` directory, prepare its journal, then invoke workflow.py stages publish, scratch, evaluate. Review canonical destination collisions first: historical outputs must be archived/preserved, never overwritten. Publish reuses the original scientific runs; scratch generates only missing coverage; evaluate uses shared contracts/TrackEval. Run select_tracker.py only for the preplanned three tracker settings. `selected_tracker_settings.json` records the accepted explicit override; base YAML remains unchanged for detector cache identity.

To replay the selected YOLO tracker into a FRESH file using the actual production CLI:

```powershell
.venv/Scripts/python.exe scripts/run_yolo.py track --split val --match 117093 --frameset win60s --tracker-settings {rel}/selected_tracker_settings.json --output {rel}/manual_selected_replay.csv
```

The equivalent production replay was executed and byte-matched the accepted selected tracks. Never run the held-out downloader or train/test commands as part of this demonstration. Review MANUAL_ACCEPTANCE_CHECKLIST, KNOWN_LIMITATIONS and DISCUSSION_GUIDE before the instructor demonstration. Presentation remains pending.
''')
    doc('FINAL_SCIENTIFIC_RESULTS.md','# Accepted validation results\n\nAll rows use117093; E3 is598eval10frames, E4 is1500win60sframes; values are fractions, not percentages.\n\n## E1 sampled patches\n\n'+mdtable(e1,['method','accuracy','sensitivity','specificity','f1','roc_auc','pr_auc'])+'\n\n## E2 all proposals\n\n'+mdtable(e2,['method','proposals','recall50','recall30','f1'])+'\n\nStarting proposal recall50='+str(data['starting_E2']['breakdown']['proposal_recall_at_iou']['0.5'])+' with '+str(data['starting_E2']['counts']['proposals'])+' proposals; classification and coverage are distinct.\n\n## E3 detections\n\n'+mdtable(e3,['method','precision','recall','f1','ap50','ap50_95'])+'\n\n## E4 tracking\n\n'+mdtable(e4,['method','hota','deta','assa','idf1','mota','idsw'])+'\n\nSelected scratch: '+data['selected_scratch_detector']+'. Selected real YOLO tracker: '+data['tracker_selection']['selected']+'. Controlled ByteTrack rows retain starting settings. Selection uses the preplanned three-candidate HOTA criterion, not unique-ID count. Full settings and variants are recorded in accepted_results.json and variants/tracker/.\n\nInterpretation: YOLO improves detection recall/AP. CNN narrowly leads scratch F1, while HOG has higher precision. YOLO tracking has higher HOTA/IDF1 but lower MOTA than selected scratch tracking; these metrics penalize different errors. No independent test/generalization or medical fatigue claim is made.')
    doc('FINAL_EXPERIMENT_MATRIX.md','''# Executed experiment matrix

| Experiment | Inputs/window | Methods | Evidence/status |
| --- | --- | --- | --- |
| E1 |704saved validation patches | Auxiliary majority, calibrated HOG/SVM, small CNN | tables/E1_accepted.csv; executed; no retraining |
| E2 |598eval10frames, all scored proposals | HOG/CNN selected candidate; starting HOG proposal variant | tables/E2_accepted.csv; proposal coverage at.5/.3; variants/starting/metrics.json |
| E3 |Same598frames and GT | Rules/HOG/CNN/YOLO | tables/E3_accepted.csv; floor.05/reporting.50; AP50/AP50:95 |
| E4 controlled |1500contiguous frames | Oracle/noisy with IoU/Hungarian and starting ByteTrack | tables/E4_accepted.csv; TrackEval; seed42/drop.10/jitter2px |
| E4 real |1500contiguous frames | Selected CNN/IoU and YOLO/ByteTrack | Canonical serialized detections; production replay; selected conservative override |
| Tracker selection |Same serialized YOLO source | Existing starting/conservative/long_buffer | variants/tracker; max HOTA, IDF1/MOTA tiebreak; supplementary validation selection |
| Optional cross-tracker grid |Not required | Deferred | Not executed; mandatory work prioritized |
| Held-out/Phase3 |Prohibited |Not executed | No128057/132831/117092 data access |

Check coverage.json, source sidecars, model_references.json, selected_tracker_settings.json and accepted_results.json for exact identity/configuration. Runtime scopes differ; no end-to-end speed ranking is asserted.
''')
    doc('MEMBER4_INTEGRATION_RESULTS.md','''# Member4 integration results

Verified clean788592f at start; fd2c05a corrective behavior preserved. Completed incoming config mapping/schema/path checks, enabled-split protection before resolution, absolute registry/model paths, contained artifact components, automatic read-only-root enforcement and validated frameset values. All nine APIs now function with the ignored local approved-only registry. No machine path was hardcoded into reusable modules.

check_data all now means enabled train+val, never every registry entry; test requests reject before loading sources. Missing files count as failures, integer frames parse exactly, missing-video handles release and Windows console output uses ASCII. Existing report destinations are versioned on collision. Downloader reviewed read-only; see the explicit inherited print-only dry-run test deviation in FINAL_VALIDATION_AND_TESTS.

Visualization now releases frame/writer handles, checks coverage/opening, preserves source FPS/stride/aspect and rejects video overwrite. CLI explicit figure roots remain contained, overlays use metadata windows instead of max prediction frame, comparison accepts selected pipelines and duplicate main dispatch was removed. Shared EDA, stages, overlays and selected comparison commands were executed; logs and shared_visuals contain outputs.

No earlier finite-value/int64/raw-score/split/protocol/HOTA/noisy-boundary correction was reverted. Aggregate patch-score and box-sample schemas remain1.0. Five Phase3 placeholder tests remain intentionally excluded by scope. The shared notebook is now a real executed cache-first scientific workflow, not fixture fallbacks.
''')
    doc('FINAL_VALIDATION_AND_TESTS.md',f'''# Final validation evidence

Authorized full suite: **{tests}**. Actual full command: `{default_command}`. Initial default run: `{summary(run/'final_tests.log')}`. Its extra pass was the inherited print-only downloader test; final authorized selection excludes that prohibited operation explicitly. HOG numerical assertion is unchanged and remains visible as a failure. Five skips are future Phase3; data/GPU tests are separately selected.

Bounded real/GPU/artifact suite: **{real}**. Exact final command is in accepted_real_command.txt. It covers config paths, actual CUDA/YOLO, saved cache/notebook/video, source parity, proposal recall, all accepted table contracts and cache-only analyzer behavior. An initial harness guessed kind from directory names and failed for versioned tracker variants; fixed to read authoritative sidecar kind. No assertion was weakened. An earlier broad data test selection was interrupted to avoid redundant six-clip full decoding; no pass is claimed for that interrupted attempt.

Production smoke: eight actual CLIs returned0 (smoke/results.json), including data check, score export, three scratch methods, bounded YOLO, IoU and E3. Full real tracking ran production CLIs; explicit selected-tracker replay was byte-identical (production_replay_verification.json). Shared visualization EDA/stage/overlay/comparison commands produced real files. Node syntax checks pass for both browser scripts.

GUI HTTP/API: ten views and all protected upload guards passed (gui_api_verification.json). Original/GT skip inference; cached views do not replay association; exact seeking and unavailable-frame behavior tested. This is NOT browser visual testing. ComputerUse ended because it could not confidently determine the current Windows browser URL; no alternative UI automation bypass was attempted. Play/pause/zoom/layout acceptance remains manual.

Notebook: {notebook['executed_cells']}/{notebook['code_cells']} code cells executed; {notebook['sections']} sections; zero error outputs. Actual saved outputs and config hash checked. Video:150decoded frames,25FPS,2048x1080two-pane panorama; no inference rerun. Report formats: HTML/Markdown; PDF conversion blocked by available rendering setup, so no PDF/layout certification claim.

Boundary deviation: inherited test_config::test_07_download_dry_run invoked the downloader's PRINT-ONLY main path during initial tests, contrary to the instruction not to execute it. Static review confirms it returned before network/data-file operations; no download or protected-data access occurred. Final suite explicitly deselects it; original logs retained.
''')
    actions=[('Launch application','Run the START_HERE command and open localhost8765','Integrated validation tab and accepted E1–E4 tables'),('Open117093','Select Accepted video analyzer','Same approved validation video; no test selector'),('Original','Choose Original, frame0','Unannotated panorama; scope says no inference'),('Ground Truth','Choose GT','GT boxes on the same frame; no inference'),('Pitch and foreground','Choose Pitch/Foreground','Actual selected scratch masks, native aspect'),('Proposals','Choose Candidate proposals','All candidate regions; no classifier invocation'),('Rules','Choose Rules detections','Cached selected-config rules boxes'),('HOG/SVM','Choose HOG/SVM detections','Cached calibrated-model boxes; HOG caveat disclosed'),('CNN','Choose CNN detections','Cached trained-CNN boxes'),('YOLO','Choose Tiled YOLO detections','Native-offset boxes from verified cache'),('Identical frames','Keep frame0 and switch detectors','Original frame remains identical; counts may differ'),('IoU tracking','Choose Tracking and selected CNN','IDs from accepted contiguous replay; confidence0shows all emitted rows'),('ByteTrack','Choose Tracking and YOLO','Accepted conservative tracker; IDs; explicit settings in manifest'),('Detection errors','Open representative_detection_error.png/crop','Red missed GT, green predictions; frame1410 recorded'),('Tracking errors','Open tracking_timeline/transition figures','Qualitative matched-ID changes, not official IDSW counts'),('E1','Review patch table','Majority75%accuracy/zero recall; identical704samples'),('E2','Review coverage/counts','Starting vs selected; all598frame proposals'),('E3','Review detector table','Same598frames; source AP/floor/threshold visible'),('E4','Review tracker table','1500frames; real/controlled rows; HOTA/IDF1/MOTA/IDSW distinct'),('Playback and controls','Play/pause, seek/rewind, switch models, cancel, click image zoom','Correct selected frame/preset; missing cache frames show unavailable'),('Notebook','Open executed HTML/notebook','16executed code cells,32sections; actual cached figures/tables'),('Report','Open report HTML/Markdown','Complete scientific narrative and matching result tables'),('Consistency','Compare E3/E4 across GUI/notebook/report/CSV','Same accepted values and config hash'),('Video','Open synchronized_comparison.mp4','150frames/6seconds, same frame IDs, correct aspect'),('Uploads','Use live scratch analyzer with an ordinary supported clip','Max512MiB; GT disabled without annotations; no YOLO cache falsely applied')]
    actions=[(a,b,c.replace('16executed code cells,32sections',f"{notebook['executed_cells']} executed code cells, {notebook['sections']} sections")) for a,b,c in actions]
    doc('MANUAL_ACCEPTANCE_CHECKLIST.md','# Manual acceptance checklist\n\nBrowser visual testing could not be completed by automation. For each step, expected content is below. If anything differs, stop that view and send the step number, screenshot, frame/model/view, browser console error and latest gui_logs operation record. Do not send private videos unnecessarily.\n\n'+mdtable([dict(step=i+1,action=a,expected=b,check=c) for i,(a,b,c) in enumerate(actions)],['step','action','expected','check'])+'\n\nLinks: START_HERE.md, accepted_results.json, figures/, tables/, videos/, notebook_verification.json. GUI thresholds affect display only; tracking confidence filtering does not rerun association. Unsupported cache frames/combinations must never show fabricated results.')
    doc('DISCUSSION_GUIDE.md','''# Instructor discussion guide

We detect22on-pitch players and assign temporary image-space identities. Full-pitch SoccerTrack footage exposes tiny distant players and long trajectories. Match-level training/validation separation prevents mixing adjacent frames of the same match across the training boundary; validation selection still limits generalization.

The scratch pipeline teaches background subtraction, pitch masking, morphology, components and perspective rules. HOG summarizes gradients; a calibrated SVM separates player/background patches. The CNN learns visual features through weighted BCE and Adam. YOLO is a pretrained COCO-person comparison, not newly trained here. Native overlapping tiles retain detail; coordinate translation and global NMS merge duplicate boxes.

Hungarian tracking assigns boxes by gated IoU and ages unmatched tracks. ByteTrack also uses lower-confidence observations. A tracker can preserve IDs while admitting false tracks, so HOTA, IDF1, MOTA and ID switches must be read together. Count of IDs is not IDSW.

E1 measures sampled-patch discrimination, E2 proposal coverage/natural imbalance, E3 full-frame detection and E4 continuous identities. Majority accuracy75%hides zero player recall. Patch AP differs from detection AP. Proposal recall limits any classifier using those candidates.

Actual findings: selected proposal coverage increases0.5412to0.7083. YOLO E3F1 is0.8162; CNN0.7174 narrowly exceeds HOG0.7093. Selected YOLO tracker HOTA0.4321/IDF1 0.4327 exceeds CNN/IoU0.2551/0.2680; its MOTA0.3255 is below0.4058. Explain the precision/recall and identity/false-track trade-offs rather than claiming one universal winner. Existing candidate tracker comparison selects conservative settings; no new model search.

Limitations: repeated negative patches, limited sampled training, one validation match, offline future-frame background, tiny/occluded players, temporary IDs, runtime scope differences and unavailable OpenCV numerical reference. No held-out/medical-fatigue claim. Phase3 may investigate calibrated workload proxies, explainability and separately authorized generalization testing.

Demonstration: open integrated results, then Original/GT on frame0, switch CNN/HOG/YOLO on the same frame, inspect mask/proposals/tiles, show both trackers, compare E3/E4, show a missed-player crop and six-second synchronized video, then the executed notebook/report. Presentation slides remain a separate task.
''')
    doc('PRESENTATION_EVIDENCE_INVENTORY.md','''# Evidence for later presentation — no slides created

Use tables/E1_accepted.csv through E4_accepted.csv as the only final numbers. Recommended figures: figures/eda/boxes_per_match.png, box_size_hist.png; patch_examples.png; imbalance.png; hog_features.png; cnn_architecture.png; cnn_training.png; scratch_stages.png; proposal_comparison.png; detection_comparison.png; eval/pr_detection_val.png; tracking_comparison.png; synchronized_frame_0.png; representative_detection_error.png; missed_player_crop.png; tracking_timeline.png; tracking_transition.png. YOLO historical intermediates: figures/yolo_stages/{tiles,tile_local,pre_nms,post_nms}.jpg, verified by yolo_stage_provenance.json.

Video: videos/synchronized_comparison.mp4,6seconds at25FPS,selected scratch versus YOLO on identical frames. Tracking figures use accepted outputs; qualitative transition labels are not official CLEAR switch-event annotations. Training curves are historical CNN evidence; no YOLO training curve is claimed.

Suggested discussion order: question/data/split -> scratch and pretrained designs -> shared E1–E4 -> fair comparison/errors -> limitations/Phase3. Final PPT intentionally pending and must not be marked complete.
''')
    doc('KNOWN_LIMITATIONS.md','''# Disclosed limitations

- OpenCV5.0 lacks HOGDescriptor in both project and existing external reference environment; base Python lacks OpenCV. No compatible available reference found. Numerical parity is UNKNOWN; custom features and original assertion remain unchanged. Separate isolated reference proposal: demos/member1/HOG_REFERENCE_ISSUE.md. No package installation/retraining authorized or performed.
- One validation match and validation-driven selection; no held-out128057/132831 or disabled117092 access. Duplicate proposal negatives retain historical weighting;3432/704nominal versus2782/580unique patches. Background preparation is offline/noncausal.
- Proposal coverage limits scratch detection. Tiny/occluded players and association fragmentation remain substantial. Selected YOLO tracking has higher HOTA/IDF1 but lower MOTA than selected scratch. Results are not physiological fatigue predictions.
- Runtime scopes differ: shared scratch processing allocation, whole union wall time, historical warm YOLO8.1282FPS and association-only time cannot be treated as one end-to-end speed ranking.
- Browser visual QA was blocked by ComputerUse URL-policy enforcement. Ten API views and protected guards passed; play/pause/seek/layout/zoom remain for manual acceptance. No browser testing success is claimed.
- Report supplied as complete HTML and editable Markdown. PDF/DOCX not generated: unsigned PowerShell script execution was blocked, main environment lacks a COM Python bridge, no policy/dependency change attempted. User may print/export the HTML manually and inspect pagination.
- Initial inherited test called downloader's print-only dry-run despite the explicit do-not-execute instruction. It returned before network/data reads; no files downloaded. Final authorized suite deselects that operation. This deviation is recorded, not concealed.
- Presentation is required academically but intentionally excluded/pending. Phase3 is not implemented. Full academic submission readiness is not claimed.
''')
    academic=[('Preprocessing/split/leakage','Complete and verified','Shared parser/config/guards; approved train/val only'),('Imbalance and trained baseline','Complete with disclosed limitation','Existing SVM/CNN; duplicate weights disclosed'),('Architectures/training/settings','Complete and verified','Actual checkpoints/curves/configuration'),('E1–E4/comparison/interpretation','Complete with disclosed limitation','Accepted artifacts; validation selection and HOG caveat'),('Reference paper comparison','Complete with disclosed limitation','Primary HTML verified; GSR/BAS not numerically comparable'),('Report/notebook','Complete and verified','HTML/Markdown; executed notebook/HTML; PDF not required by available source convention'),('GUI browser visual acceptance','Blocked with specific evidence','ComputerUse could not verify current browser URL; API tests pass'),('Presentation','Blocked with specific evidence','Intentionally pending by user instruction'),('Cross-tracker extra grid','Optional and not required','Not executed'),('Held-out/explainability/workload','Deferred to Phase3','No execution')]
    doc('FINAL_IMPLEMENTATION_HANDOFF.md',f'''# Final implementation handoff

Verdict: **READY FOR MANUAL ACCEPTANCE WITH EXPLICIT NON-BLOCKING LIMITATIONS; PRESENTATION PENDING**. Start with START_HERE.md. Scientific validation runs are complete; manual GUI/report visual review and presentation remain. No installation, scientific retraining, repeated full YOLO inference or Git write occurred.

Member4 infrastructure was adapted without reverting prior corrections. Production data/scoring/Rules/HOG/CNN/YOLO/tracking/evaluation/visualization commands executed. Missing scratch inference used one1948-frame shared union; verified YOLO cache reused. Full E1–E4 and planned tracker selection completed. Existing models/patches/history preserved. GUI extends the existing Member1 application, with cache-backed actual artifacts and separately labeled live/historical scratch views. Browser acceptance is manual due tool-policy block.

Notebook: `{ROOT/source}`; executed `{run/'phase2_baseline_executed.ipynb'}`; HTML `{run/'phase2_baseline_executed.html'}`. Verification: {notebook['executed_cells']}executed code cells,{notebook['sections']}sections,zero errors. Report `{run/'report/phase2_report.html'}`; editable Markdown `{run/'report/phase2_report.md'}` and `{ROOT/'docs/phase2/report/phase2_report.md'}`. Figures `{run/'figures'}`; video `{run/'videos/synchronized_comparison.mp4'}`; official tables `{run/'tables'}`. Models/configs: model_references.json, selected_tracker_settings.json and configurations/. Logs/verification live alongside this file.

Final tests: {tests}; bounded real/GPU: {real}. Sole HOG API failure unchanged; five Phase3 skips. See FINAL_VALIDATION_AND_TESTS for the downloader dry-run deviation, justified deselections and all limits. Exact changed inventory/manual staging in MANUAL_STAGING.md. No commit/push performed; user owns Git.

## Academic completion matrix

'''+mdtable([dict(requirement=a,status=b,evidence=c) for a,b,c in academic],['requirement','status','evidence']))
    # Preserve base configuration files and a precisely labeled final reproduction source snapshot.
    for name in ('project','scratch','yolo','evaluation'):copy_new(ROOT/f'configs/{name}.yaml',run/f'configurations/{name}.yaml')
    tracked=set(json.loads((run/'tracked_before.json').read_text()))
    changed=sorted(set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines()))
    (run/'changed_files.txt').write_text('\n'.join(changed)+'\n')
    for name in changed:
        if (ROOT/name).is_file():copy_new(ROOT/name,run/'source_snapshot'/name)
    doc('MANUAL_STAGING.md','# Manual staging — NOT EXECUTED\n\nReview the exact'+str(len(changed))+'source/test/document paths below. No outputs/data/models/registry/temporary files should be staged.\n\n```powershell\ngit diff --check\ngit add -- '+' '.join('"'+p+'"' for p in changed)+'\ngit diff --cached --name-only\ngit diff --cached --check\n```\n\nDo not stage outputs/, data/, configs/data_sources.yaml or weights. No commit/push command is included.\n\n'+ '\n'.join('- '+p for p in changed))
    before=read(run/'preservation_before.json');historical={p:h for p,h in before.items() if p.startswith('outputs/') or p.startswith('docs/phase1/')}
    deviations=[p for p,h in historical.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    hogold=ast.parse(git('show','HEAD:tests/test_patches_features.py'));hognew=ast.parse((ROOT/'tests/test_patches_features.py').read_text(encoding='utf-8-sig'))
    node=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='test_05'),include_attributes=False)
    preservation=dict(checked_utc=datetime.now(timezone.utc).isoformat(),historical_files_checked=len(historical),changed_historical=deviations,
        hog_numerical_assertion_unchanged=node(hogold)==node(hognew),git_head=git('rev-parse','HEAD'),branch=git('branch','--show-current'),index_unchanged=not bool(git('diff','--cached','--name-only')),
        features_unchanged=sha(ROOT/'src/soccertrack/features.py')==before['src/soccertrack/features.py'],classifiers_unchanged=sha(ROOT/'src/soccertrack/classifiers.py')==before['src/soccertrack/classifiers.py'])
    write(run/'preservation_after.json',preservation)
    if deviations or not all(preservation[k] for k in ('hog_numerical_assertion_unchanged','index_unchanged','features_unchanged','classifiers_unchanged')):raise RuntimeError('Preservation check failed')
    doc('ARTIFACT_PRESERVATION_REPORT.md',f'''# Preservation report

{len(historical)}recorded historical output/Phase1files hashed before/after; **zero changed/missing**. Original Member1/2 models, datasets, predictions, executed notebooks, videos, manifests and previous audits preserved. Existing external data/weights used read-only; no protected match data access. Source features/classifiers and original HOG numerical assertion unchanged. Source modifications are authorized integration changes, recorded separately.

HEAD remains{preservation['git_head']}, branch{preservation['branch']}; Git index unchanged. No Git writes performed. Current accepted run contains new/versioned outputs and immutable source references. Starting/alternative tracker outputs preserved while conservative selection was published canonically. No old scientific result was relabeled as a new run.

ARTIFACT_MANIFEST records accepted hashes/config/weights and final reproduction-source hashes. source_snapshot is the final reproducible integration source; stage logs/journal retain execution chronology, including metadata/visualization corrections. No claim is made that every historical source file was replaced by this snapshot. External drives were not exhaustively enumerated/hashed. See preservation_after.json for machine-readable evidence and FINAL_VALIDATION_AND_TESTS for the print-only dry-run boundary deviation.
''')
    write(ROOT/'outputs/phase2_final/latest.json',{'path':rel})
    if not (run/'accepted_real_command.txt').exists():
        (run/'accepted_real_command.txt').write_text(".venv/Scripts/python.exe -B -m pytest tests/test_environment.py tests/test_config.py tests/test_detector_yolo.py tests/test_member2_scientific.py tests/test_segmentation_proposals.py::test_data_01 tests/test_member1_corrective.py::test_real_same_frame_predictions_and_track_inputs_match_saved_candidate tests/test_phase2_acceptance.py -m 'data or gpu' -q --basetemp="+rel+"/accepted_real_tmp -o cache_dir="+rel+"/pytest_cache --tb=short\n")
    required=['START_HERE.md','FINAL_IMPLEMENTATION_HANDOFF.md','MEMBER4_INTEGRATION_RESULTS.md','FINAL_EXPERIMENT_MATRIX.md','FINAL_SCIENTIFIC_RESULTS.md','FINAL_VALIDATION_AND_TESTS.md','MANUAL_ACCEPTANCE_CHECKLIST.md','DISCUSSION_GUIDE.md','PRESENTATION_EVIDENCE_INVENTORY.md','ARTIFACT_PRESERVATION_REPORT.md','KNOWN_LIMITATIONS.md','MANUAL_STAGING.md','EXECUTION_JOURNAL.md']
    if not all((run/n).is_file() for n in required):raise RuntimeError('Required handoff missing')
    files={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file() and p.name!='ARTIFACT_MANIFEST.json' and not any(s in p.parts for s in ('gui_logs','notebook_runtime','mpl','yolo','pytest_cache')) and '_tmp' not in str(p)}
    write(run/'ARTIFACT_MANIFEST.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),git_head=preservation['git_head'],config_sha256=data['config_sha256'],
          models=read(run/'model_references.json'),selected_tracker_settings=data['tracker_selection'],files=files,changed_files=changed,
          verdict='READY FOR MANUAL ACCEPTANCE WITH EXPLICIT NON-BLOCKING LIMITATIONS; PRESENTATION PENDING'))
    print('Final package complete:',run,'changed source files:',len(changed),'historical files preserved:',len(historical))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();package(a.run)
