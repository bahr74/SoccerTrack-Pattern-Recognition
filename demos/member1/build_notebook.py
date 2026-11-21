"""Build/execute the isolated educational notebook without global kernel changes."""
from pathlib import Path
import argparse,json,os,sys
import nbformat as nb
from scientific_workflow import ROOT,BASE,HEADINGS,write_json

def build():
    cells=[nb.v4.new_markdown_cell('# Member 1 — Scratch Football Detection and Tracking\n\nEducational local validation workflow. Official benchmark integration and final held-out testing are **NOT RUN**. Outputs are recomputed from saved artifacts and approved data. Run top-to-bottom; training is off by default.')]
    sections=[
('We detect players using hand-built segmentation/proposals, then rules, HOG/SVM or a small CNN, and associate detections using IoU/Hungarian matching. Member 2 owns pretrained YOLO; it is outside this notebook. The aim is to understand errors at every stage, not claim benchmark completion.', '''import os, sys
from pathlib import Path
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'src/soccertrack').is_dir())
sys.path.insert(0, str(root/'demos/member1'))
import numpy as np, pandas as pd, cv2
from IPython.display import display, Image, Markdown
import scientific_workflow as w
from notebook_support import context
from soccertrack import data_io, segmentation, proposals, patches, features, detector_scratch, tracker_iou
SMOKE = os.environ.get('MEMBER1_NOTEBOOK_MODE', 'cached') == 'smoke'
RETRAIN = False  # explicit opt-in; opening the notebook never retrains
ctx = context(smoke=SMOKE)
run, m = ctx['run'], ctx['m']
print(m['status']); print('Mode:', 'SYNTHETIC SMOKE' if SMOKE else 'APPROVED REAL VALIDATION')
def figure(name):
    override=(ctx.get('supplement') or {}).get('presentation_override',{}).get(name)
    path=w.ROOT/override if override else run/name
    if path.is_file(): display(Image(filename=str(path), width=1000))
    else: print('NOT GENERATED:', name)
def metric_table(values):
    return pd.DataFrame(values).T.drop(columns=['confusion_matrix'], errors='ignore')
print('Pipeline: frame → masks → proposals → scores → NMS → local track IDs')''', 'The visible status applies to every following result. Synthetic smoke results demonstrate execution only; they do not support claims about SoccerTrack performance.'),
('SoccerTrack uses full-resolution BGR video and zero-based MOT annotations. The approved split is five training matches and validation 117093. Optional 117092 is disabled; held-out matches are never opened here. Cached mode requires the completed local run, registered sources and its trusted checkpoints.', '''display(pd.DataFrame([{k:d.get(k) for k in ['match','split','frames_annotated','before_rows','after_rows']} for d in m['dataset']]))
display(pd.DataFrame([d['video_header'] for d in m['dataset']]))
print('Train sample matches:', sorted(set(ctx['train']['match'])))
print('Validation sample matches:', sorted(set(ctx['val']['match'])))
print('Final test:', m['final_test'])
figure('stages/gt_overlay.png')''', 'Header frame counts describe container metadata; a full decoded-count audit of all six videos was not performed. The overlay demonstrates actual annotation alignment on a sequentially decoded frame.'),
('Sampling uses explicit sequential indices, never timestamp seeking. Boxes remain full-frame xywh; only patch cropping changes color to RGB. Train/validation matches are disjoint in the real experiment. Label-free background estimation spans the validation clip, so the pipeline is offline and not a causal streaming benchmark.', '''if not SMOKE:
    assert set(ctx['train']['match']).isdisjoint(set(ctx['val']['match']))
assert ctx['train']['X'].shape[1:] == (64,32,3)
assert ctx['val']['frame'].dtype == np.int32
print('Sample plan:', m['frame_plan'])
print('First patch full-frame xywh:', ctx['val']['box'][0])
print('Source shape:', ctx['original'].shape, 'BGR uint8; patch shape:', ctx['val']['X'][0].shape, 'RGB uint8')
print('Excluded frames:', m.get('excluded_frames', []))''', 'Real training covers 39 successfully sampled frames; validation patches cover eight other frames. Detection report frames 0–19 and configuration-development frames 200/250/300 are separate from patch sampling, but share the validation match and are not a final test.'),
('The parser rejects missing/nonfinite or malformed input. Geometry cleaning counts duplicate removals, clipping and degenerates, preserves IDs, and audits expected players per frame. The local helper does not certify the shared schema. A synthetic dirty annotation example below exercises cleaning without altering real source data.', '''display(pd.DataFrame([dict(match=d['match'], before=d['before_rows'], after=d['after_rows'], missing=d.get('raw_missing_values'), **d['issues']) for d in m['dataset']]))
raw = pd.DataFrame([[0,1,-3,2,20,40,-1,-1,-1,-1], [0,1,-3,2,20,40,-1,-1,-1,-1], [0,2,30,20,0,10,-1,-1,-1,-1]], columns=['frame','id','x','y','w','h','c7','c8','c9','c10'])
cleaned, issues = data_io.clean_mot_geometry(raw, (100,80), expected_boxes_per_frame=1)
print('SYNTHETIC cleaning example:', issues); display(cleaned)
assert len(cleaned)==1 and cleaned.iloc[0].x==0''', 'The real labels had no detected duplicate, degenerate or out-of-bounds records. This is evidence of local geometry quality, not proof of perfect identity annotation or full-video alignment.'),
('EDA covers cleaned annotations across enabled matches and the sampled patch class balance. Width, height and area expose perspective; foot-point density and track lengths reveal where the detector must operate. Appearance examples are actual sampled patches.', '''figure('figures/eda_boxes.png')
figure('figures/eda_spatial_tracks.png')
figure('figures/class_balance.png')
figure('figures/train_patches.png')
from provenance import patch_summary
display(pd.DataFrame({name:patch_summary(ctx[name]) for name in ['train','val']}).T)
print('Patch counts:', {name:dict(total=len(ctx[name]['y']), positives=int(ctx[name]['y'].sum()), negatives=int((ctx[name]['y']==0).sum())) for name in ['train','val']})''', 'All approved real frames have 22 annotation rows, so raw GT row counts alone conceal detection difficulty. Proposal negatives are sampled with replacement. The historical real train set has 3,432 total / 2,782 distinct patches (650 repeated copies); validation has 704 / 580 (124 repeated copies). Proposal negatives have only 130 distinct train patches out of 780, and 36 distinct validation patches out of 160. These are within-split weights, not demonstrated train/validation leakage. Sparse frame sampling and correlation limit generalization conclusions.'),
('Median backgrounds and HSV pitch masks operate at the configured scale. Color differencing followed by opening/closing yields foreground. Cropping pads context and converts BGR to RGB exactly once. The size model and CNN mean/std use training data only; validation backgrounds use images without GT.', '''pitch = segmentation.pitch_mask(ctx['bg'], ctx['runtime'], ctx['item']['match'])
p, raw_foreground, clean_foreground = detector_scratch.prepare_frame_proposals(ctx['original'], 0, ctx['bg'], pitch, ctx['size'], ctx['runtime'])
print('Background shape:', ctx['bg'].shape, 'pitch pixels:', int(pitch.sum()))
print('Raw / cleaned foreground pixels:', int(raw_foreground.sum()), int(clean_foreground.sum()))
print('CNN train-only channel mean/std:', ctx['models']['cnn'].mean, ctx['models']['cnn'].std)
for name in ['02_background.png','03_pitch.png','04_foreground.png','05_morphology.png']: figure('stages/'+name)''', 'A player present in many background samples may disappear from foreground. Morphology can merge nearby players; both effects constrain proposal recall before classification.'),
('Connected components become full-frame proposals after size filtering and wide-component splitting. Expected height is fitted on training GT. Local proposal recall tests whether any proposal overlaps each GT box at IoU 0.5; it is not official detector AP.', '''selected = m['paired_detection'][m['recommended_configuration']]
print('Size model:', m['size_model'])
print('Frame-zero proposals:', len(p)); display(p.head())
print('Local proposal recall @0.5:', selected['proposal_recall_at_0_5'])
figure('stages/06_proposals.png'); figure('stages/failure_example.png')''', 'Missing proposal coverage cannot be repaired by a better patch classifier. The red missed-GT and magenta unmatched-detection examples expose localization and foreground failures.'),
('The explicitly named patch baseline is sklearn DummyClassifier(strategy="most_frequent"), fitted on training labels and evaluated on exactly the same validation patches as SVM/CNN. Rules are a separate, untrained detection baseline. Tracking has no official baseline metric until the approved evaluator exists.', '''from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(ctx['train']['y']),1)),ctx['train']['y'])
baseline_scores=baseline.predict_proba(np.zeros((len(ctx['val']['y']),1)))[:,1]
display(metric_table({'majority_baseline':w.patch_metrics(ctx['val']['y'],baseline_scores)}))
print('Rule DETECTION baseline:'); display(metric_table({'rules': selected['models']['rules']}))''', 'With the configured 3:1 negative ratio, a majority classifier gets 75% accuracy while detecting no positives at 0.5. Accuracy alone would therefore overstate usefulness.'),
('HOG uses 8×8 cells, 2×2 blocks, nine unsigned orientation bins, bilinear voting and L2-Hys. A 64×32 patch yields 756 features. There is no data-fitted HOG transform. OpenCV 5 removed core HOGDescriptor; the reference test is still failing and numerical parity is unverified.', '''H=features.hog_features(ctx['val']['X'][:4],ctx['cfg'])
print('HOG matrix:', H.shape, 'finite:', np.isfinite(H).all())
assert H.shape==(4,756)
print('Core OpenCV HOGDescriptor available:', hasattr(cv2,'HOGDescriptor'))
figure('figures/hog_example.png')''', 'The manual descriptor is a useful independent feature implementation, but these plots do not establish OpenCV equivalence. In particular, Gaussian weighting within reference blocks needs verification in an approved compatible environment.'),
('HOG/SVM uses a balanced linear margin classifier with train-only probability-calibration folds. Validation selects C from the configured grid. The CNN uses Conv-BN-ReLU-MaxPool ×3 (16/32/64 channels), global average pooling, dropout and a binary linear logit.', '''print('SVM configuration:', ctx['cfg']['svm'])
print(ctx['models']['cnn'].model)
print('CNN parameters:', sum(p.numel() for p in ctx['models']['cnn'].model.parameters()))
print('SVM selected C:', ctx['models']['hog_svm'].selected_C)
print('Checkpoint inventory:'); display(pd.DataFrame(m['models']).T)''', 'Both classifiers use the same train/validation patch arrays. Validation-based C/epoch selection means reported patch scores are development estimates, not untouched test performance.'),
('Existing checkpoints load by default and their SHA-256 hashes are checked. CNN optimization uses weighted BCE, Adam, augmentation and validation AP early stopping; train-only statistics prevent normalization leakage. Retraining is an explicit opt-in and creates a new run.', '''if RETRAIN:
    if SMOKE: raise ValueError('Use smoke context for its explicit one-epoch training')
    new_run=w.run_experiment(epochs=8)
    print('New run:',new_run)
else:
    print('Retraining OFF. Loaded:', run)
print('Seed:',m['seed']); display(pd.DataFrame(w.read_json(run/'svm_search.json')))
display(pd.DataFrame(w.read_json(run/'training_history.json')))
figure('figures/training_curves.png')''', 'Training loss includes augmentation and class weighting; validation loss uses fixed examples, so the curves are not directly interchangeable. The selected checkpoint can precede the last epoch.'),
('Patch metrics are accuracy, precision, recall, F1, average precision (PR-AUC convention here), ROC-AUC and confusion matrices. Full-frame detection reports local greedy precision/recall/F1 separately. Official AP/mAP, HOTA and IDF1 require Member 3’s approved evaluators; final held-out testing is NOT RUN.', '''fresh={name:w.patch_metrics(ctx['val']['y'],model.predict_proba(ctx['val']['X'])) for name,model in ctx['models'].items()}
fresh['majority_baseline']=w.patch_metrics(ctx['val']['y'],baseline_scores)
for name,values in fresh.items():
    assert np.isclose(values['f1'],m['classification_metrics'][name]['f1'])
display(metric_table(fresh)); figure('figures/confusion_matrices.png'); figure('figures/patch_pr_curves.png')
print('AP/mAP, HOTA, IDF1: NOT RUN; final held-out test:',m['final_test'])''', 'Recomputed scores verify checkpoint reuse. These are sampled validation patches with an artificial class balance, so their precision and AP cannot be substituted for full-frame detector performance.'),
('The detector crops candidates, scores them, retains a low score floor and applies torchvision NMS. The reporting operating threshold is 0.5. Boxes retain full-frame coordinates. GT overlays and FP/FN diagnostics use the same zero-based frame selection.', '''scored, detections=detector_scratch.classify_candidates(ctx['original'],p,ctx['runtime'],ctx['models']['hog_svm'])
print('Proposals / scored / post-NMS / score>=0.5:',len(p),len(scored),len(detections),int((detections.score>=.5).sum()))
print('Detector settings:',ctx['runtime']['detector'])
figure('stages/08_classification.png'); figure('stages/09_detections.png'); figure('stages/gt_overlay.png'); figure('stages/failure_example.png')
display(metric_table(selected['models']))''', 'The local greedy matching diagnostic is not the official evaluator. Low proposal overlap, classification thresholding and NMS interactions can each cause missed players.'),
('Each consecutive frame produces a gated IoU cost matrix. Hungarian assignment minimizes cost, then tracks are confirmed, aged and removed according to configured rules. This tracker has no appearance or motion model; occlusion, crossing and missed detections can fragment identities.', '''from scipy.optimize import linear_sum_assignment
small=p[['x','y','w','h']].to_numpy()[:5]
overlap=tracker_iou._iou(small,small)
display(pd.DataFrame(overlap))
if len(small):
    gate=ctx['runtime']['tracker_iou']['iou_gate']; rr,cc=linear_sum_assignment(np.where(overlap>=gate,1-overlap,1e6))
    print('Frame-zero self-overlap assignment demonstration:',list(zip(rr,cc)))
tracks=pd.read_csv(run/'tracking_local.csv'); display(tracks.head())
print('Actual tracking summary:',m['tracking'])
if (run/'supplemental_comparison.json').is_file():
    print('Local identity diagnostic:',w.read_json(run/'supplemental_comparison.json')['identity_diagnostic'])
figure('stages/10_tracking.png'); figure('figures/trajectories.png')''', 'The small IoU matrix is an explicitly labeled self-overlap mechanics demonstration; the saved track table comes from consecutive real detections. No HOTA/IDF1 or identity-switch rate is claimed.'),
('Compare patch classifiers only on identical patches, and detector variants only on identical frames, thresholds, GT and background. Candidate segmentation settings were selected using separate development frames. Error categories are overlap-based diagnostics, not causal ground truth. Tracking association quality remains unquantified.', '''display(metric_table(m['classification_metrics']))
for name,value in m['paired_detection'].items():
    print(name,'proposal recall',value['proposal_recall_at_0_5'])
    display(metric_table(value['models']))
if (run/'supplemental_comparison.json').is_file():
    paired=w.read_json(run/'supplemental_comparison.json')
    print(paired['conditions'])
    for label in ['earlier_pilot_models','expanded_models']:
        print(label); display(metric_table(paired[label]['models']))
if ctx.get('supplement'):
    print(ctx['supplement']['sensitivity_status'])
    display(metric_table(ctx['supplement']['unique_patch_sensitivity_metrics']))
    print(ctx['supplement']['future_experiment'])
print('Overlap-based error attribution:'); display(pd.DataFrame(selected.get('error_counts',{})).fillna(0))
figure('figures/detection_thresholds.png'); figure('stages/failure_example.png')
print('Limitations:',m['limitations'])''', 'An improvement must be supported by the paired table; a higher CNN patch score does not guarantee a higher detector score. Earlier pilot comparisons change training coverage, while the baseline/candidate comparison holds models fixed. Unique-patch metrics change sample weights and class balance; they are sensitivity analyses, not replacements for the historical results or official metrics. Any sampling improvement requires a versioned experiment; existing arrays/checkpoints remain unchanged.'),
('All figures, local CSV diagnostics, configuration, seeds, checkpoint hashes and sample plans reside under the versioned run directory. Shared production exports remain gated by M0. Reproduce the expanded run with the commands below; use explicit smoke mode when approved data is unavailable.', '''print('Run directory:', run)
print('Checkpoints:',m['models'])
print('Configuration SHA-256:',w.sha(run/'configuration.json'))
if ctx.get('supplement'):
    print('RETROSPECTIVE metadata clarification:',ctx['supplement']['metadata_discrepancies'])
    print('Resolved replay settings:',ctx['supplement']['resolved_settings'])
    print('Validate only (no training): .venv\\Scripts\\python.exe demos/member1/scientific_workflow.py --replay-settings '+ctx['supplement']['resolved_settings'])
print('Final test:',m['final_test'])
print('Reproduce: .venv\\Scripts\\python.exe demos/member1/scientific_workflow.py --run --epochs 8')
print('Notebook: .venv\\Scripts\\python.exe demos/member1/build_notebook.py --execute')
print('Smoke: .venv\\Scripts\\python.exe demos/member1/build_notebook.py --execute --smoke')
print('GUI: .venv\\Scripts\\python.exe demos/member1/app.py --port 8765')
print('Next steps: approved HOG reference environment, shared validators/evaluators, broader frozen validation, identity-error audit.')
display(Markdown('**Conclusions:** Use the measured tables above. This is a reproducible local development study, not a certified final benchmark.'))''', 'Source notebooks keep outputs cleared for team review; an executed copy and HTML report are saved under ignored outputs. No raw data, weights or caches should be staged into Git.')]
    for i,(intro,code,interpretation) in enumerate(sections):
        cells.extend([nb.v4.new_markdown_cell(f'## {i+1}. {HEADINGS[i]}\n\n{intro}'),nb.v4.new_code_cell(code),nb.v4.new_markdown_cell('**Interpretation.** '+interpretation)])
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Member 1 project .venv','language':'python','name':'member1-local'},'language_info':{'name':'python','version':'3.12'}})
    path=ROOT/'demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb'; path.parent.mkdir(exist_ok=True); nb.write(notebook,path)
    return path

def execute(path,smoke=False):
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    from datetime import datetime,timezone
    target=BASE/'notebook_executions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); target.mkdir(parents=True)
    kernels=BASE/'jupyter_data/kernels/member1-local'; kernels.mkdir(parents=True,exist_ok=True)
    write_json(kernels/'kernel.json',{'argv':[str(ROOT/'.venv/Scripts/python.exe'),'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Member 1 project .venv','language':'python'})
    os.environ['JUPYTER_PATH']=str(BASE/'jupyter_data'); os.environ['JUPYTER_RUNTIME_DIR']=str(BASE/'jupyter_runtime'); os.environ['IPYTHONDIR']=str(BASE/'ipython')
    os.environ['MEMBER1_NOTEBOOK_MODE']='smoke' if smoke else 'cached'
    notebook=nb.read(path,as_version=4); client=NotebookClient(notebook,timeout=600,kernel_name='member1-local',resources={'metadata':{'path':str(ROOT)}})
    try: client.execute()
    finally: nb.write(notebook,target/path.name)
    html,_=HTMLExporter().from_notebook_node(notebook); (target/'member1_workflow.html').write_text(html,encoding='utf-8')
    audit=[]
    for i,title in enumerate(HEADINGS):
        markdown=notebook.cells[1+i*3]; code=notebook.cells[2+i*3]
        audit.append({'section':title,'heading_ok':markdown.source.startswith(f'## {i+1}. {title}'),'executed':code.execution_count is not None,'outputs':len(code.outputs),'errors':[o.get('ename') for o in code.outputs if o.output_type=='error']})
    write_json(target/'section_audit.json',audit); print(target,flush=True)
    assert all(a['heading_ok'] and a['executed'] and not a['errors'] for a in audit)
    return target

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute',action='store_true'); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    path=build(); print(path,flush=True)
    if args.execute: execute(path,args.smoke)
