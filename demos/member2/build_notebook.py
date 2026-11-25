"""Generate and execute the Member 2 cached-results notebook without expensive inference."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import nbformat as nbf


def build(run,execute=True):
    from demos.member2.safety import require_absent
    run=require_absent(run,['member2_yolo_pipeline_experiments_source.ipynb',
        'member2_yolo_pipeline_experiments_executed.ipynb','member2_yolo_pipeline_experiments.html','notebook_execution.json'])
    run=Path(run).resolve()
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text))
    def code(text): cells.append(nbf.v4.new_code_cell(text))
    md('''# Member 2: Pretrained YOLO + ByteTrack experiments
SBES 426 — Medical Pattern Recognition | Phase 2 | Validation match **117093**.

**Provisional local scientific validation. Official E3/E4, AP, HOTA, IDF1 and MOTA are blocked by unimplemented shared dependencies. Held-out testing and fine-tuning were not run.**

This notebook loads verified caches by default. It does not train, download or rerun detection. Detection estimates boxes and confidence scores; tracking associates observations across time. Neither directly estimates physiological fatigue.''')
    code('''from pathlib import Path
import json, hashlib, os
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/yolo.yaml').is_file())
RUN = ROOT / "'''+run.relative_to(ROOT).as_posix()+'''"
manifest = json.loads((RUN/'manifest.json').read_text())
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
hashes = json.loads((RUN/'artifact_hashes.json').read_text())
assert all(sha(RUN/p)==value for p,value in hashes.items()), 'Saved artifact changed'
assert manifest['split']=='val' and manifest['match']=='117093'
weights = Path('F:/Football-Fatigue-Project/models/yolov8n.pt')
assert weights.is_file() and sha(weights)==manifest['weights_sha256']
def show(relative, width=1000): display(Image(filename=str(RUN/relative), width=width))
display(pd.DataFrame([{'run_id':manifest['run_id'], 'status':manifest['status'], 'verified_artifacts':len(hashes), 'device':manifest['device']}]))''')
    md('''## 1. Problem definition and objectives
Detect the annotated football players as full-frame top-left `(x,y,w,h)` boxes, then attach temporary positive IDs to their observations. Confidence measures the detector's person score, not identity certainty. Ground truth supplies annotated player positions and original player IDs for evaluation, never for suppressing detections at inference.

We test zero-shot transfer from COCO people to tiny panoramic players. Pipeline B is independent of Member 1's proposal-dependent HOG/SVM and small-CNN detectors.''')
    code("display(pd.DataFrame([{'detector':'yolov8n_coco','tracker':'bytetrack','class':'COCO person (0)','training_this_run':'none','target':'22 annotated players/frame'}]))")
    md('''## 2. Dataset description and ground truth
SoccerTrack v2 release v1.2, revision `325bd4c57ee8e135ab652f3921d274526ada496b`. Each validation frame has 22 annotated player boxes. Referees, staff and spectators can be genuine people without player annotations. A person detection may therefore be counted as a false positive by the player benchmark.

Raw MOT is already 0-based. Sequential `data_io.iter_frames` avoids the clip's container-time offset. We reuse Member 1's raw parser/geometry cleaner; official GT conversion remains gated by Member 3 validation.''')
    code("display(pd.DataFrame([manifest['video_info']]))\ndisplay(pd.DataFrame([manifest['gt_cleaning_issues']]))\nshow('figures/gt_frame_0.jpg')")
    md('''## 3. Data preparation and approved split
Train: 118575, 118576, 118577, 118578, 128058. Validation: 117093. Optional 117092 stays disabled. Held-out 128057 and 132831 were not accessed.

There is no new YOLO training, normalization fitted to GT, class resampling or augmentation in this inference workflow. Ultralytics letterboxes each tile and normalizes model input internally. Controlled experiments use identical development frames, disjoint from the required reporting frames. The entire validation match remains development data at the match level.''')
    code("development = manifest['development_frame_ids']\nreport = manifest['required_union_frame_ids']\nassert not set(development) & set(report)\ndisplay(pd.DataFrame([{'development_frames':len(development),'required_union_frames':len(report),'overlap':len(set(development)&set(report))}]))\nprint('Development frame IDs:', development)")
    md('''## 4. Pretrained model and weight provenance
Only the existing absolute local `yolov8n.pt` is loaded, after SHA-256 verification. No models, datasets or dependencies are downloaded. `YOLO_OFFLINE=true` and `YOLO_AUTOINSTALL=false` suppress automatic network checks/installations. The model is COCO-pretrained; this team did not train it in Phase 2. A trusted local serialized checkpoint is required; do not load arbitrary uploaded checkpoints.''')
    code("display(pd.DataFrame([{'weights_sha256':manifest['weights_sha256'],'git_commit':manifest['git_commit'],'dataset_revision':manifest['dataset_revision'],'created_utc':manifest['created_utc']}]))\ndisplay(pd.DataFrame(json.loads((RUN/'environment.json').read_text())['packages'].items(),columns=['package','version']))")
    md('''## 5. YOLOv8n architecture and training mechanism
The loaded network has Conv/C2f backbone blocks and SPPF context pooling. Upsampling, concatenation and C2f blocks aggregate features across scales; the Detect head predicts at strides 8, 16 and 32. It is anchor-free with respect to predefined anchor-box shapes: predictions use grid anchor points and learned box-distance distributions.

The installed `v8DetectionLoss` uses BCEWithLogits classification, CIoU box loss and Distribution Focal Loss for distance-bin regression; TaskAlignedAssigner assigns training targets. These describe the pretrained architecture's training mechanism, **not losses measured in this experiment**. No optimizer, epochs, training curve or new loss values are claimed. Native small-object information can still be lost through feature downsampling.''')
    code("architecture = json.loads((RUN/'architecture.json').read_text())\ndisplay(pd.DataFrame([{k:architecture[k] for k in ['parameters','head','nc','reg_max','strides']}]))\ndisplay(pd.DataFrame(architecture['layers']))")
    md('''## 6. Small-object detection challenges
A 20-pixel-wide player shrinks to about 6.25 pixels when a 4096-pixel panorama is reduced to 1280 pixels. A 1280-wide native tile avoids that horizontal shrink at `imgsz=1280`. Smaller tiles add inference calls and can truncate players; overlap helps preserve border context. Tiling cannot recover genuinely missing visual detail, heavy occlusion or annotation mismatch.''')
    code("display(pd.DataFrame([{'source_player_width':20,'whole_frame_width':4096,'input_width':1280,'resized_player_width':20*1280/4096}]))\nshow('required_validation/stages/original.jpg')")
    md('''## 7. Tiling and tile coverage
Tiles cover every native pixel, with the final tile aligned to the image edge. The configured overlap is a minimum; edge alignment may create larger overlap. Synthetic coverage tests include 4096×1080, 3840×1504, exact tile sizes and frames smaller than a tile. Invalid dimensions and overlap are rejected.''')
    code("geometry = json.loads((RUN/'required_validation/stages/geometry.json').read_text())\ndisplay(pd.DataFrame(geometry['tiles'],columns=['x0','y0','x1','y1']))\nshow('required_validation/stages/tiles.jpg')\nshow('figures/overlap_crop.jpg',width=560)")
    md('''## 8. Tile-local YOLO detection
BGR uint8 crops are passed together to the installed `YOLO.predict`, requesting only class 0. Per-tile NMS IoU is 0.5 and confidence floor is 0.05. Predictions are already expressed in each crop's native pixel coordinates after Ultralytics reverses its letterbox transform. Keeping the low floor supports both reporting-threshold curves and ByteTrack's low-score association stage.''')
    code("display(pd.DataFrame([manifest['selected_configuration']['inference']]))\nshow('required_validation/stages/tile_local.jpg',width=850)")
    md('''## 9. Global coordinates and non-maximum suppression
For tile origin `(x0,y0)`, add `[x0,y0,x0,y0]` to local xyxy, clip to the full image, then perform torchvision global NMS at IoU 0.5. Convert retained boxes to top-left xywh. Invalid/degenerate boxes are dropped. Global NMS removes repeated views across overlap; it may also suppress distinct nearby players with overlapping boxes.

Pitch filtering is off: no approved shared pitch-mask artifact was available. No ground-truth-derived inference mask is used.''')
    code("display(pd.DataFrame([{'pre_global_nms':len(geometry['pre_nms']),'post_global_nms':len(geometry['post_nms']),'pitch_filter':manifest['pitch_filter']}]))\nshow('required_validation/stages/pre_nms.jpg')\nshow('required_validation/stages/post_nms.jpg')")
    md('''## 10. Detection baseline and starting configuration
The historical-style reference uses 1280×1080 tiles, overlap 64, `imgsz=960`, reporting threshold 0.40 and retained floor 0.05. Its historical pitch mask is unavailable, so this is a documented partial reproduction. Historical project numbers are not reused as current measurements. Each candidate changes one factor relative to the 1280-input baseline where practical.''')
    code("grid = pd.read_csv(RUN/'development_results.csv')\nreference = pd.read_csv(RUN/'reference_threshold_comparison.csv')\ndisplay(reference)\nprint('All rows are provisional local validation counts, not official AP.')")
    md('''## 11. Validation configuration and threshold experiments
Candidate tile widths 960/1280, inference sizes 960/1280 and overlaps 64/128 are paired on the same 33 frames. The small bounded study is not an exhaustive grid. Selection maximizes local F1 at IoU 0.5, with runtime breaking ties. Global NMS is fixed; pitch on/off is blocked by the missing approved mask.

The final reporting threshold differs from the retained score floor, per-tile/global NMS IoU, evaluation IoU, and ByteTrack high/low thresholds. These control separate decisions. The reporting threshold is not applied to tracker input.''')
    code("display(pd.read_csv(RUN/'candidate_optima.csv')[['configuration','threshold','precision','recall','f1','fps']])\nprint('Selected:',manifest['selected_candidate'],manifest['selected_configuration']['operating_threshold'])\nshow('figures/controlled_experiments.png')\nshow('figures/threshold_analysis.png')")
    md('''## 12. ByteTrack architecture and tracking pipeline
Top-left xywh becomes centre-based xywh before calling the installed BYTETracker. Its Kalman state tracks centre x/y, aspect ratio and height with velocities. High-score association uses IoU distance with optional score fusion and LAP assignment; remaining tracked objects associate with low-score detections using IoU-only cost (fixed second-stage limit 0.5 in this installed version). Lost tracks may reactivate; stale lost tracks are removed after the configured buffer.

New tracks require `new_track_thresh`; later arrivals require confirmation before being returned. Update is called on every consecutive frame, including empty frames. State resets per independent sequence. Sparse eval10 is rejected. Raw associated detection boxes/scores are written through returned full-input association indices. A Kalman-output mode is also tested. There is no appearance ReID or guaranteed identity through crossings.''')
    code("tracking = pd.DataFrame(manifest['tracking_diagnostics'])\ndisplay(tracking.drop(columns=['settings']))\nshow('figures/tracking_diagnostics.png')\nshow('figures/trajectories.png')")
    md('''## 13. Detection and tracking evaluation
Local detection diagnostics use score-ordered greedy one-to-one matching at IoU 0.5, checked against the shared hand-computed fixture. TP is a matched prediction; FP is unmatched; FN is unmatched GT. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). Object detection has no well-defined background-box TN count, so specificity, accuracy and ROC-AUC are not reported here.

Member3 validators/evaluators and oracle/noisy-GT producers are implemented. Official execution awaits Member4 integration and canonical cache migration/replay. AP/mAP/HOTA/IDF1/MOTA/official ID switches are **NOT RUN** for this local experiment. Track counts and lengths are diagnostics, not substitute tracking metrics. Full eval10/win60s coverage includes empty-prediction frames.''')
    code("display(pd.DataFrame(manifest['required_detection_results']))\nfor name,expected in [('eval10',598),('win60s',1500)]:\n    coverage=json.loads((RUN/f'required_validation/{name}/coverage.json').read_text())\n    assert coverage['decoded_frames']==expected\n    print(name,expected,'frames; coverage verified')\nprint('Official E3:',manifest['official_e3'])\nprint('Official E4:',manifest['official_e4'])")
    md('''## 14. Comparative results and runtime
Compare candidates only on the identical development frames at the same threshold or clearly stated individual optimum. Report-frame results use the selected fixed threshold. This within-match separation reduces direct frame reuse but is not an unbiased final test.

Processing time includes tile copying, Ultralytics preprocessing/inference/local NMS and global merge, with CUDA synchronization; initial warmup is included. Model load is separate. The combined cache's end-to-end timing includes sequential decoding and stage export; output CSV writing and later visualization are excluded. Framesets share a union decode pass, so subset processing FPS is measured directly, while subset-specific end-to-end FPS is not claimed.

Member 1 learns patch classification behind handcrafted proposals; YOLO jointly detects boxes and person classes using pretrained features. Member 1 uses IoU/Hungarian links; ByteTrack adds Kalman motion and low-score recovery. There is no certified paired A/B performance comparison yet.''')
    code("display(reference[['configuration','threshold','precision','recall','f1','fps']])\nwarmed = pd.read_csv(RUN/'warmed_runtime_probe.csv').groupby('configuration').processing_seconds.agg(['mean','median'])\nwarmed['processing_fps'] = 1/warmed['mean']\ndisplay(warmed)\nprint('Separate eight-frame warmed probe: setup/decode excluded. Full-run timings include cold starts and possible GPU smoke-test contention.')\ndisplay(pd.DataFrame([{'union_frames':manifest['decoded_union_frames'],'decode_included_seconds':manifest['union_decode_included_seconds'],'decode_included_fps':manifest['union_decode_included_fps']}]))")
    md('''## 15. Error analysis and scientific limitations
Green boxes are local TP, red FP and blue FN at the fixed selected threshold. Inspect native-pixel crops before attributing failures. Unannotated people are a plausible source of FP, not a universal explanation. Small distant players, border fragments, score calibration and crowd overlap can cause misses or poor localization. Tracking can fragment after long gaps or change identities during crossings.

The visual examples are selected illustrations, not a prevalence estimate. In `failure_0_fp_0.jpg`, a visible touchline person holds a board and is outside the player GT: this supports the person-versus-player mismatch interpretation for that example. `failure_1000_fn_6.jpg` shows a missed white-kit player near a blue-kit player; overlap is a plausible difficulty, without proving the detector's internal cause. Predicted trajectories remain image-space paths, not calibrated speed, workload or physiological fatigue.''')
    code("show('figures/errors_frame_0.jpg')\nshow('figures/errors_frame_500.jpg')\nshow('figures/errors_frame_1000.jpg')\nfor path in sorted((RUN/'figures').glob('failure_0_*.jpg'))[:6]:\n    display(Image(filename=str(path),width=220))\nprint('Annotated video:',RUN/'annotated_validation.mp4')\ndisplay(pd.DataFrame([json.loads((RUN/'video_verification.json').read_text())]))")
    md('''## 16. Conclusions and reproducibility
The local pipeline has synthetic correctness tests and real validation evidence. Official integration remains blocked by Members 3/4; held-out M6 is prohibited. Phase 3 may fine-tune on approved train tiles, improve tiny-object recall, calibrate pitch filtering and assess appearance-based association. No Phase 3 work is executed here.

To reproduce explicitly, from the repository root with the existing environment:

```powershell
.venv/Scripts/python.exe demos/member2/scientific_workflow.py --output outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/runtime_probe.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/build_artifacts.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
.venv/Scripts/python.exe demos/member2/build_notebook.py outputs/member2_scientific/runs/NEW_UNIQUE_RUN
```

No fresh directory is overwritten. Pipeline sources, input hashes, full configurations, seeds, frame IDs, device, versions, measured timings and artifact hashes accompany the run. See Member 2 README and scientific handoff for the exact shared interfaces still required.''')
    code("print('Verified run:',manifest['run_id'])\nprint('No training, downloads or held-out testing performed by notebook execution.')\nprint('Verdict: PARTIAL — SHARED DEPENDENCIES BLOCKED')")
    notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'member2','display_name':'Member 2 project venv','language':'python'},'language_info':{'name':'python','version':'3.12'}})
    source=run/'member2_yolo_pipeline_experiments_source.ipynb'
    source.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,source)
    if execute:
        task=run/'notebook_runtime'
        for key,folder in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','config'),('JUPYTER_DATA_DIR','data'),('JUPYTER_RUNTIME_DIR','runtime'),('TEMP','temp'),('TMP','temp')]:
            target=task/folder
            target.mkdir(parents=True,exist_ok=True)
            os.environ[key]=str(target)
        os.environ['JUPYTER_PATH']=str(task/'jupyter')
        kernel=task/'jupyter/kernels/member2'
        kernel.mkdir(parents=True,exist_ok=True)
        (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Member 2 project venv','language':'python'}))
        from nbclient import NotebookClient
        from nbconvert import HTMLExporter
        executed=NotebookClient(notebook,timeout=180,kernel_name='member2',resources={'metadata':{'path':str(ROOT)}}).execute()
        target=run/'member2_yolo_pipeline_experiments_executed.ipynb'
        nbf.write(executed,target)
        exporter=HTMLExporter()
        exporter.embed_images=True
        body,_=exporter.from_notebook_node(executed)
        (run/'member2_yolo_pipeline_experiments.html').write_text(body,encoding='utf-8')
        saved=nbf.read(target,as_version=4)
        code_cells=[c for c in saved.cells if c.cell_type=='code']
        errors=[o for c in code_cells for o in c.outputs if o.output_type=='error']
        assert not errors and all(c.execution_count is not None and c.outputs for c in code_cells)
        evidence={'executed_code_cells':len(code_cells),'markdown_cells':sum(c.cell_type=='markdown' for c in saved.cells),
                  'error_outputs':len(errors),'all_code_cells_have_outputs':True,'source_outputs_cleared':True,
                  'sections':[c.source.splitlines()[0] for c in saved.cells if c.cell_type=='markdown'],
                  'executed_sha256':sha_file(target),'html_sha256':sha_file(run/'member2_yolo_pipeline_experiments.html')}
        (run/'notebook_execution.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(evidence,indent=2))
    return source

def sha_file(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--no-execute',action='store_true')
    args=parser.parse_args()
    build(args.run,not args.no_execute)
