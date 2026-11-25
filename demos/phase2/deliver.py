"""Build the unified scientific deliverables from accepted artifacts, without inference/training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,os,sys,html,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from soccertrack import config,contracts as C,features,metrics as M,data_io,visualize
from demos.phase2.workflow import read,write,copy_new,journal,M1,sha

def visuals(run,data):
    folder=run/'figures';folder.mkdir(exist_ok=True)
    with np.load(M1/'patches_val.npz',allow_pickle=False) as a:
        X=a['X'][:];y=a['y'][:]
    fig,axes=plt.subplots(2,8,figsize=(12,4))
    for row,label in enumerate([1,0]):
        for ax,i in zip(axes[row],np.flatnonzero(y==label)[:8]):ax.imshow(X[i]);ax.set_title('Player' if label else 'Negative');ax.axis('off')
    fig.suptitle('Real validation patches — sampled RGB64x32, historical duplicates disclosed');fig.tight_layout();fig.savefig(folder/'patch_examples.png',dpi=140);plt.close(fig)
    sc=config.load_config()['scratch'];H=features.hog_features(X[:1],sc)
    fig,axes=plt.subplots(1,2,figsize=(6,4));axes[0].imshow(X[0]);axes[1].imshow(features.hog_glyph(H[0],sc),cmap='gray')
    for ax in axes:ax.axis('off')
    fig.suptitle('Custom756-D HOG; OpenCV parity unverified');fig.savefig(folder/'hog_features.png',dpi=140);plt.close(fig)
    history=read(M1/'training_history.json');fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].plot(history['epoch'],history['train_loss'],label='Train weighted BCE');axes[0].plot(history['epoch'],history['val_loss'],label='Validation weighted BCE');axes[0].legend();axes[0].set(xlabel='Epoch',ylabel='Loss')
    axes[1].plot(history['epoch'],history['val_pr_auc'],marker='o');axes[1].axvline(7,color='gray',linestyle='--');axes[1].set(xlabel='Epoch',ylabel='Validation patch AP',title='Historical selection:epoch7')
    fig.tight_layout();fig.savefig(folder/'cnn_training.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,2.2));ax.axis('off')
    boxes=['RGB64x32','Conv16 + Pool','Conv32 + Pool','Conv64 + Pool','Global pooling','Dropout + Linear1']
    for i,label in enumerate(boxes):
        x=.01+i*.166;ax.text(x,.5,label,ha='left',va='center',fontsize=9,bbox=dict(boxstyle='round',facecolor='#e6eef4',edgecolor='#334f66'))
        if i<5:ax.annotate('',xy=(x+.16,.5),xytext=(x+.13,.5),arrowprops=dict(arrowstyle='->',color='#334f66'))
    ax.set_title('Small CNN ·23,873 trainable parameters · weighted BCEWithLogits · Adam');fig.tight_layout();fig.savefig(folder/'cnn_architecture.png',dpi=150);plt.close(fig)
    for task,keys,name in [('E3',['f1','ap50','ap50_95'],'detection_comparison.png'),('E4',['hota','idf1','mota'],'tracking_comparison.png')]:
        docs=[d for d in data['results'] if d['eval_id']==task];fig,ax=plt.subplots(figsize=(12,4.6))
        for i,key in enumerate(keys):ax.bar(np.arange(len(docs))+i*.25,[d['metrics'][key] for d in docs],.25,label=key)
        ax.set_xticks(np.arange(len(docs))+.25,[d['method'] for d in docs],rotation=18,ha='right');ax.set_ylabel('Fraction; MOTA may be negative');ax.legend();ax.set_title(task+' ·same approved validation protocol');fig.tight_layout();fig.savefig(folder/name,dpi=150);plt.close(fig)
    improved=next(d for d in data['results'] if d['eval_id']=='E2' and d['method']=='scratch_hog_svm');start=data['starting_E2']
    fig,axes=plt.subplots(1,2,figsize=(9,3.6))
    axes[0].bar(['Starting','Selected'],[d['breakdown']['proposal_recall_at_iou']['0.5'] for d in [start,improved]]);axes[0].set(ylabel='GT coverage at IoU0.5',ylim=(0,1))
    axes[1].bar(['Starting','Selected'],[d['counts']['proposals'] for d in [start,improved]]);axes[1].set(ylabel='All598-frame proposals');fig.tight_layout();fig.savefig(folder/'proposal_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3.4));ax.bar(['Positive train','Negative train','Positive val','Negative val'],[858,2574,176,528],color=['#226e91','#c9953f']*2);ax.set_ylabel('Nominal patch rows');ax.set_title('1:3 patch sampling — repeated proposal negatives retained');fig.tight_layout();fig.savefig(folder/'imbalance.png',dpi=140);plt.close(fig)
    # One shared decode produces synchronized real comparison images/video.
    best=data['selected_scratch_detector'];gt=data_io.load_gt_tracks('117093',config.load_data_sources());clip=config.match_paths('117093',config.load_data_sources()).clip
    tables={}
    for method in ('scratch_mwb0','scratch_hog_svm','scratch_cnn','yolov8n_coco'):
        item=data['artifacts'][f'detections/{method}/win60s'];tables[method]=C.read_table(run/item['path'],'detections')
    videos=run/'videos';videos.mkdir(exist_ok=True);path=videos/'synchronized_comparison.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(2048,1080))
    if not writer.isOpened():raise OSError('Comparison writer unavailable')
    iterator=data_io.iter_frames(clip,range(150));seen=[]
    try:
        for f,image in iterator:
            panes=[]
            for method in (best,'yolov8n_coco'):
                d=tables[method];d=d[(d.frame==f)&(d.score>=.5)]
                pane=visualize.draw_boxes(image,d,f);cv2.putText(pane,f'{method} |117093|frame{f}|threshold.50|config {data["config_sha256"][:10]}',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3)
                panes.append(cv2.resize(pane,(2048,540)))
            writer.write(np.vstack(panes));seen.append(f)
            if f in (0,100):
                allpanes=[]
                for method,d in tables.items():
                    p=visualize.draw_boxes(image,d[(d.frame==f)&(d.score>=.5)],f);cv2.putText(p,method,(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3);allpanes.append(cv2.resize(p,(1280,338)))
                cv2.imwrite(str(folder/f'synchronized_frame_{f}.png'),np.vstack(allpanes))
    finally:writer.release();iterator.close()
    cap=cv2.VideoCapture(str(path));count=0
    try:
        while True:
            ok,image=cap.read()
            if not ok:break
            if image.shape[:2]!=(1080,2048):raise ValueError('Video aspect mismatch')
            count+=1
    finally:cap.release()
    if count!=150 or seen!=list(range(150)):raise ValueError('Comparison video incomplete')
    write(run/'video_verification.json',dict(frames=count,fps=25.,size=[2048,1080],each_pane_aspect='4096:1080',source_frames=seen,detectors=[best,'yolov8n_coco'],source='accepted canonical detections, no inference'))
    # Representative detection error case chosen using actual shared matching.
    errorrows=[]
    for method,d in tables.items():
        for f in range(0,1500,10):
            truth=gt[gt.frame==f];pred=d[(d.frame==f)&(d.score>=.5)]
            match=M.match_frame(truth[['x','y','w','h']].to_numpy(),pred[['x','y','w','h']].to_numpy(),pred.score.to_numpy(),.5)
            errorrows.append(dict(method=method,frame=f,tp=int(match['tp_flags'].sum()),fp=int((~match['tp_flags']).sum()),fn=int((~match['gt_matched']).sum())))
    pd.DataFrame(errorrows).to_csv(run/'tables/error_examples.csv',index=False)
    worst=max((r for r in errorrows if r['method']==best),key=lambda r:r['fn']+r['fp'])
    f=worst['frame'];image=next(data_io.iter_frames(clip,[f]))[1];d=tables[best];pred=d[(d.frame==f)&(d.score>=.5)];truth=gt[gt.frame==f]
    match=M.match_frame(truth[['x','y','w','h']].to_numpy(),pred[['x','y','w','h']].to_numpy(),pred.score.to_numpy(),.5)
    missed=truth.loc[~match['gt_matched']]
    panel=visualize.draw_boxes(image,pred,f)
    for row in missed.itertuples():cv2.rectangle(panel,(round(row.x),round(row.y)),(round(row.x+row.w),round(row.y+row.h)),(0,0,255),2)
    cv2.imwrite(str(folder/'representative_detection_error.png'),panel)
    if len(missed):
        row=missed.iloc[0];x,y,w,h=map(float,row[['x','y','w','h']]);crop=image[max(0,int(y)-20):min(1080,int(y+h)+20),max(0,int(x)-20):min(4096,int(x+w)+20)]
        cv2.imwrite(str(folder/'missed_player_crop.png'),cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST))
    write(run/'error_selection.json',dict(worst_example=worst,missed_boxes=missed[['x','y','w','h']].values.tolist(),legend='Green predictions; red missed GT at IoU0.5'))

def narrative(data):
    e3={d['method']:d['metrics'] for d in data['results'] if d['eval_id']=='E3'}
    e4={d['method']:d['metrics'] for d in data['results'] if d['eval_id']=='E4'}
    r0=data['starting_E2']['breakdown']['proposal_recall_at_iou']['0.5']
    r1=next(d for d in data['results'] if d['eval_id']=='E2')['breakdown']['proposal_recall_at_iou']['0.5']
    yt=e4['yolov8n_coco+bytetrack'];st=e4[data['selected_scratch_detector']+'+iou_hungarian']
    return [
    ('Project objective','We detect and track players in panoramic soccer video. Phase2 compares rule-based vision, a trained HOG/SVM, a trained small CNN and pretrained tiled YOLOv8n, using one shared validation protocol. We estimate image-space player boxes and temporary identities, not physiological fatigue.'),
    ('Reference and approved scope','We use the paper’s curated bounding-box subset for image-space MOT. GSR/BAS benchmarks differ in task and protocol; no direct leaderboard claim is made. Primary reference: https://arxiv.org/html/2508.01802v2.'),
    ('Dataset organization','A clip and ten-column MOT file are joined by match ID. Native images are4096x1080 at25FPS. Internal frames are zero-based and boxes use x,y,width,height. Rows describe the22 on-pitch players. Ball actions, jersey recognition and pitch-space reconstruction are outside the executed scope.'),
    ('Data preparation and cleaning','MOT parsing preserves integer identities, rejects malformed/nonfinite values, removes duplicates/degenerate geometry, clips valid intersections and reports issue counts. Sequential decoding supplies requested frames and verifies coverage. Source annotations and pretrained weights remain read-only.'),
    ('Exploratory data analysis','The EDA tables summarize only the five approved training matches and validation117093. Small distant players motivate native-resolution processing, perspective-aware proposals and tiled inference. Linear empirical quantiles use rank(n-1)q. EDA frame counts below distinguish header/dense-annotation agreement from the newly decoded evaluation union.'),
    ('Match split and leakage prevention','Training matches are118575,118576,118577,118578,128058; validation is117093. Protected128057/132831 and disabled117092 were not opened. Split checks precede data resolution. Selection on validation is disclosed; no independent test performance is claimed.'),
    ('Patch sampling and imbalance','Historical training contains3432 patches:858 positive and2574 negative, sampled at39 match/frame pairs; validation contains704 patches:176/528 over8frames. Crops are RGB64x32 with context. Proposal negatives repeat: unique pixels are2782/580. We preserve the fitted data rather than silently deduplicating history.'),
    ('Background and pitch segmentation','The scratch pipeline compares each frame to a label-free median background, masks the pitch in HSV/Lab processing and cleans foreground differences. Background samples span the clip, including future frames: this is an offline, noncausal method. Cached background reuse avoids unnecessary decoding.'),
    ('Morphology and candidate proposals','Thresholded differences, median filtering, opening/closing, vertical dilation and connected components form regions. A size model fitted to training data accounts for player height versus image row; filtering and wide-component splitting reduce background/merged blobs. Selected candidate settings are compared with the starting settings on identical eval10 frames.'),
    ('Rule based baseline','Shape, fill and expected perspective size produce a score without a learned classifier. This baseline tests how far domain-specific image processing can go. It is a detection baseline and must not be mislabeled as the majority patch classifier.'),
    ('HOG feature representation','Unsigned orientation votes and spatial cells form756 custom HOG features per64x32 crop. Block normalization reduces contrast dependence. The feature implementation and trained SVM are unchanged. Installed OpenCV lacks HOGDescriptor, so reference numerical parity remains unverified despite internal reproducibility.'),
    ('SVM training and calibration','Balanced LinearSVC uses the squared-hinge objective. Training-only stratified folds calibrate scores with a sigmoid; validation patch AP selects C from0.01/0.1/1/10, with historical C0.01 retained. A calibrated score ranks proposals; good sampled-patch performance alone does not establish detector recall.'),
    ('CNN architecture','Three convolution blocks (16/32/64channels), pooling, global aggregation, dropout and one binary output contain23873 trainable parameters. RGB normalization is estimated from training patches only. Learned visual features replace fixed HOG features while using the same proposal source.'),
    ('CNN optimization and historical training','The existing model was trained for8epochs with Adam (learning rate0.001, weight decay0.0001) and weighted BCEWithLogits to address1:3 imbalance. Training augmentation includes horizontal flips and brightness/contrast changes. Validation AP selected epoch7. The displayed curves are historical training evidence; no model was retrained during integration.'),
    ('Pretrained YOLO comparison','YOLOv8n is a pretrained COCO-person reference, not a locally trained Phase2 network. Its multiscale backbone/head predicts boxes and scores. The source model uses classification and box/distribution losses during pretraining; this project performs inference only and does not invent local YOLO training curves.'),
    ('Native tiling and global NMS','1280x1080 tiles with128px overlap and inference size1280 retain tiny-player detail. Tile-local boxes are translated/clipped to native coordinates and global NMS at IoU0.5 removes duplicates. Pitch filtering remains off in the accepted selected configuration. Raw cache identity/coverage was verified before shared serialization.'),
    ('IoU Hungarian tracking','IoU measures intersection area divided by union. A gated Hungarian assignment links current boxes to tracks; confirmation and ageing handle transient misses. Every contiguous frame receives an update, including no detections. Position-only association can fragment or swap identities under occlusion.'),
    ('ByteTrack association','ByteTrack uses high-confidence detections and lower-score candidates to retain tracks through uncertain observations. Detections down to the0.05 source floor remain serialized. Reported0.50 detector threshold is separate from association thresholds. Track IDs are temporary labels; unique-ID counts do not equal identity switches.'),
    ('Shared evaluation protocol','E1classifies sampled patches; E2classifies every scored proposal and reports many-to-one GT coverage; E3matches native-frame detections one-to-one in confidence order; E4evaluates continuous tracks with TrackEval. Tables are read back after2-decimal boxes/4-decimal scores, so all final metrics use the same representation.'),
    ('E1 patch classification','We compare majority negative, HOG/SVM and CNN on the same704validation patches. Accuracy, precision, recall, specificity, F1, ROC-AUC and classification AP answer different questions. Majority prediction has75%accuracy but zero positive recall, illustrating why accuracy alone is misleading.'),
    ('E2 proposal evaluation','All proposals, including those removed by downstream NMS, are retained for598eval10 frames. Coverage at IoU0.5/0.3 measures whether any candidate can localize each GT box. Classification on natural proposal imbalance is separate. Proposal recall bounds downstream recall under that candidate/localization protocol.'),
    ('E3 common frame detection','Rules, HOG/SVM, CNN and YOLO use the same598frame IDs and GT. Precision/recall/F1 use score0.50 and IoU0.5, with IoU0.3 diagnostics. AP50 and AP50:95 use all detections retained at floor0.05 and confidence ordering. Our AP is the approved all-point implementation, not an unqualified claim of identical COCO tooling.'),
    ('E4 controlled and real tracking','Oracle and seeded noisy GT isolate association behavior for both trackers. Noise drops10%of boxes and perturbs coordinates by Gaussian2px with seed42; repaired boxes stay valid. Real selected scratch+IoU and YOLO+ByteTrack use1500contiguous frames. TrackEval supplies HOTA/DetA/AssA/LocA/IDF1/MOTA/IDSW. Controlled ByteTrack rows use starting settings; the real YOLO row selects conservative settings (high/new0.4,low0.1,buffer30,match0.8) from the three preplanned candidates by HOTA, then IDF1/MOTA. Explicit tracker overrides are stored separately from the base shared configuration; no detector reinference or new grid was introduced.'),
    ('Model comparison and selection',f'The selected scratch detector is {data["selected_scratch_detector"]}, using highest E3F1 at the fixed0.50 reporting threshold on validation. This is validation selection, not an independent generalization result. We retain the existing models and selected proposal configuration instead of introducing a new optimization search.'),
    ('Error analysis','Synchronized native-frame views expose distant players, background components, merged players and localization errors. Error-example CSVs use the shared one-to-one matcher. Red boxes mark missed GT; crops magnify actual validation pixels. TrackEval summary errors are global sequence measures; a single still image does not establish an identity switch.'),
    ('Runtime and resources','Scratch methods share decoding and proposals. Per-method processing seconds include measured shared preparation plus scoring/NMS; published frameset FPS allocates union processing proportionally and excludes decode/background/setup/write. Union wall time is measured separately. Historical YOLO warm processing8.1282FPS excludes setup/decode/write. These are different scopes; no realtime or fair end-to-end speed ranking is claimed.'),
    ('Reference comparison','Paper Table3 reports GS-HOTA12.25 over20halves and10.96 on test halves. Our60second image-space HOTA differs in coordinates, attributes and sequence length. Direct numerical superiority is not claimed.'),
    ('Scientific interpretation',f"YOLO achieves E3F1 {e3['yolov8n_coco']['f1']}, versus CNN {e3['scratch_cnn']['f1']} and HOG/SVM {e3['scratch_hog_svm']['f1']} on identical frames. The scratch selection is {data['selected_scratch_detector']}; compare precision and recall as well as this narrow validation F1 difference. Proposal coverage changes from {r0} to {r1} at IoU0.5. Selected YOLO/ByteTrack gives HOTA {yt['hota']}, IDF1 {yt['idf1']} and MOTA {yt['mota']}; selected scratch/IoU gives {st['hota']}, {st['idf1']} and {st['mota']}, respectively. Different metrics measure different errors; no universal winner or independent generalization claim follows."),
    ('Limitations','One validation match, limited sampled training coverage, repeated negatives, validation-based selection, offline future-frame background, unresolved OpenCV HOG reference verification, small/occluded players, temporary tracking identities and differing runtime scopes limit conclusions. No physiological fatigue labels or held-out results support clinical or generalization claims.'),
    ('Phase 3 and presentation','Future work may improve proposals/appearance association, test explainability, perform authorized held-out assessment and investigate calibrated pitch-space workload proxies. None is implemented as Phase3 work here. The final PowerPoint is intentionally pending, so complete academic submission readiness is not claimed.'),
    ('Reproduction and artifact identity','Normal notebook execution only loads accepted saved artifacts. Reproduction uses the production config/registry, existing hash-checked models, verified YOLO migration and checkpointed missing scratch inference. EXECUTION_JOURNAL and stage.json record progress. All historical experiments remain immutable and the user manages Git.'),
    ('Conclusion','The integrated project provides reproducible validation comparisons across classical vision, trained handcrafted/learned patch classifiers, pretrained detection and two trackers. Conclusions below are derived from accepted E1–E4 artifacts. Manual review should inspect the application, synchronized video, notebook and report before presentation.')]

def build(run,make_visuals=True):
    data=read(run/'accepted_results.json')
    if make_visuals:visuals(run,data)
    import re
    def prose(text):
        parts=re.split(r'(https?://\S+)',text)
        return ''.join(part if part.startswith('http') else re.sub(r'(?<=\d)(?=[A-Za-z])|(?<=[a-z])(?=\d)',' ',part) for part in parts)
    sections=[(prose(title),prose(text)) for title,text in narrative(data)]
    result_tables={}
    for task in ('E1','E2','E3','E4'):
        docs=[d for d in data['results'] if d['eval_id']==task]
        records=[dict(method=d['method'],**d['counts'],**d['metrics']) for d in docs]
        if task=='E1':records.insert(0,dict(method='Majority negative (auxiliary)',**data['auxiliary_majority_baseline']['counts'],**data['auxiliary_majority_baseline']['metrics']))
        table=pd.DataFrame(records);table.to_csv(run/f'tables/{task}_accepted.csv',index=False);result_tables[task]=table
    model=read(run/'model_references.json');coverage=read(run/'coverage.json')
    # Editable Markdown and standalone local HTML share one generated narrative/data source.
    report=['# SoccerTrack Phase 2 Scientific Report','', 'Integrated validation and baseline comparison','', 'SBES426 Medical Pattern Recognition | Team Members1–4 |25November2026', '',
            '## Abstract','',f'We evaluated four detectors on598identical validation frames and controlled/real trackers on1500contiguous frames. Existing SVM/CNN training and verified YOLO cache were reused. Selected scratch method: {data["selected_scratch_detector"]}. Results are validation-only; custom HOG reference parity remains unverified.','']
    body=['<h1>SoccerTrack Phase 2 Scientific Report</h1><p class="subtitle">Integrated validation and baseline comparison</p><p>SBES426 Medical Pattern Recognition · Team Members1–4 ·25November2026</p><h2>Abstract</h2><p>'+html.escape(report[-2])+'</p>']
    images={5:['eda/boxes_per_match.png','eda/box_size_hist.png'],7:['patch_examples.png','imbalance.png'],8:['scratch_stages.png'],9:['proposal_comparison.png'],11:['hog_features.png'],13:['cnn_architecture.png'],14:['cnn_training.png'],16:['yolo_stages/tiles.jpg','yolo_stages/tile_local.jpg','yolo_stages/post_nms.jpg'],20:['eval/roc_classifiers_val.png'],22:['detection_comparison.png','eval/pr_detection_val.png'],23:['tracking_comparison.png'],25:['synchronized_frame_0.png','representative_detection_error.png','missed_player_crop.png','tracking_timeline.png','tracking_transition.png']}
    for index,(title,text) in enumerate(sections,1):
        report += ['## '+title,'',text,''];body += ['<h2>'+html.escape(title)+'</h2><p>'+html.escape(text)+'</p>']
        task={20:'E1',21:'E2',22:'E3',23:'E4'}.get(index)
        if task:
            chosen=result_tables[task]
            keys=['method']+({'E1':['accuracy','sensitivity','specificity','f1','roc_auc','pr_auc'],'E2':['proposals','gt','sensitivity','f1'],'E3':['tp','fp','fn','precision','recall','f1','ap50','ap50_95'],'E4':['hota','deta','assa','idf1','mota','idsw']}[task])
            table=chosen[keys];body.append(table.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0))
            fmt=lambda v:f'{v:.4f}' if isinstance(v,float) else str(v)
            report += ['| '+' | '.join(table.columns)+' |','| '+' | '.join(['---']*len(table.columns))+' |']
            report += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in table.itertuples(index=False,name=None)]
            report += ['']
            if task=='E2':
                recalls=[dict(method=d['method'],**d['breakdown']['proposal_recall_at_iou']) for d in data['results'] if d['eval_id']=='E2']
                body.append('<p>GT proposal coverage (IoU column labels):</p>'+pd.DataFrame(recalls).to_html(index=False,border=0));report += [json.dumps(recalls),'']
        for image in images.get(index,[]):
            path=run/'figures'/image
            if path.exists():
                body.append(f'<figure><img src="../figures/{html.escape(image)}"><figcaption>{html.escape(image)} ·actual accepted or explicitly historical evidence</figcaption></figure>')
                report += [f'![{image}](../figures/{image})','']
    refs='Scott et al. SoccerTrack v2, arXiv2508.01802v2, https://arxiv.org/html/2508.01802v2 ; dataset/code https://github.com/AtomScott/SoccerTrack-v2 ; Luiten et al. HOTA, IJCV2021 ; Zhang et al. ByteTrack, ECCV2022 ; Dalal and Triggs HOG, CVPR2005 ; Ultralytics installed YOLOv8 implementation. Paper abstract/HTML and official repository documentation were read online; no dataset/weight/package download.'
    report+=['## References','',refs,''];body+=['<h2>References</h2><p>'+html.escape(refs)+'</p>']
    folder=run/'report';folder.mkdir(exist_ok=True)
    text='\n'.join(report);(folder/'phase2_report.md').write_text(text,encoding='utf-8')
    style='body{font:11pt Georgia,serif;color:#151d25;max-width:1000px;margin:40px auto;line-height:1.5}h1{font-size:28pt;color:#111}h2{font-size:17pt;break-after:avoid;color:#223b52;margin-top:1.7em}p{orphans:3;widows:3}table{font:8pt Arial,sans-serif;border-collapse:collapse;width:100%;margin:16px 0;table-layout:auto}td,th{padding:6px;border:1px solid #b9c5ce;text-align:right;overflow-wrap:anywhere}td:first-child,th:first-child{text-align:left}th{background:#e9eff3}figure{margin:20px 0;break-inside:avoid}img{max-width:100%;max-height:750px;object-fit:contain}figcaption{font:9pt Arial;color:#455665}.subtitle{font-size:17pt}@page{size:A4;margin:18mm}@media print{body{margin:0;max-width:none}h1{margin-top:25mm}table{font-size:7pt}img{max-height:190mm}}'
    (folder/'phase2_report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>SoccerTrack Phase 2 Scientific Report</title><style>'+style+'</style><body>'+''.join(body)+'</body></html>',encoding='utf-8')
    canonical=ROOT/'docs/phase2/report/phase2_report.md';canonical.parent.mkdir(parents=True,exist_ok=True);canonical.write_text(text,encoding='utf-8')
    notebook(run,data,sections)
    journal(run,'Unified report, figures/video and notebook built and executed from accepted artifacts')

def notebook(run,data,sections):
    import nbformat as nbf
    cells=[nbf.v4.new_markdown_cell('# SoccerTrack Phase 2 Scientific Workflow\nIntegrated scratch and pretrained player detection and tracking. Cache-first execution; no training, full inference or held-out access. All results are validation117093.'),
      nbf.v4.new_code_cell("from pathlib import Path\nimport json,os,numpy as np,pandas as pd\nfrom IPython.display import display,Image,Markdown\nROOT=Path.cwd()\nRUN=ROOT/os.environ.get('SOCCERTRACK_PHASE2_RUN',"+repr(run.relative_to(ROOT).as_posix())+")\ndata=json.loads((RUN/'accepted_results.json').read_text())\nprint('Accepted config:',data['config_sha256'])\nprint('Cache-first execution: no fitting or detector inference')\ndef picture(name):\n    display(Image(filename=str(RUN/'figures'/name)))")]
    for index,(title,text) in enumerate(sections,1):
        cells.append(nbf.v4.new_markdown_cell(f'## {index} {title}\n\n{text}'))
        code=None
        if index==4:code="eda=json.loads((RUN/'eda/stats.json').read_text()); display(pd.DataFrame(eda['matches']).T[['split','boxes','frames_annotated','ids']])"
        elif index==5:code="picture('eda/boxes_per_match.png'); picture('eda/box_size_hist.png')"
        elif index==7:code="refs=json.loads((RUN/'model_references.json').read_text());display(pd.DataFrame(refs['dataset']).T);picture('patch_examples.png');picture('imbalance.png')"
        elif index==8:code="picture('scratch_stages.png');print('Real frame0: raw foreground and morphology are distinct saved stages; no illustrative masks substituted')"
        elif index==9:code="picture('proposal_comparison.png')"
        elif index==11:code="picture('hog_features.png');print('756 custom features; unavailable OpenCV reference API remains disclosed')"
        elif index==13:code="picture('cnn_architecture.png');print('23873 trainable parameters; existing checkpoint, no retraining')"
        elif index==14:code="picture('cnn_training.png');history=json.loads((ROOT/'outputs/member1_scientific/runs/expanded_20261120T044321992233Z/training_history.json').read_text());display(pd.DataFrame(history))"
        elif index==16:code="arch=json.loads((ROOT/'outputs/member2_scientific/runs/member2_validation_20261125/architecture.json').read_text());print('Actual pretrained architecture:',{k:arch.get(k) for k in ['parameters','strides','head','nc']});picture('yolo_stages/tiles.jpg');picture('yolo_stages/tile_local.jpg');picture('yolo_stages/post_nms.jpg');print('Historical full-precision intermediate visuals at the selected configuration; official rounded outputs are evaluated separately')"
        elif index in (20,21,22,23):
            task={20:'E1',21:'E2',22:'E3',23:'E4'}[index]
            code=f"table=pd.read_csv(RUN/'tables/{task}_accepted.csv');display(table)\nassert len(table)>0\nprint('Loaded accepted {task} table; frame protocol recorded in each Metrics JSON')"
            if task=='E1':code+="\npicture('eval/roc_classifiers_val.png')\nfrom soccertrack.metrics import binary_classification_metrics\nscores=pd.read_csv(ROOT/'outputs/scores/scratch_hog_svm/val/patches.csv')\ncheck=binary_classification_metrics(scores.y_true.to_numpy(),scores.score.to_numpy(),.5)\nassert abs(check['metrics']['f1']-data['results'][0]['metrics']['f1'])<.0001\nprint('Independent saved-score recomputation agrees')"
            if task=='E2':code+="\ndisplay(pd.DataFrame([dict(method=d['method'],**d['breakdown']['proposal_recall_at_iou']) for d in data['results'] if d['eval_id']=='E2']))"
            if task=='E3':code+="\npicture('detection_comparison.png');picture('eval/pr_detection_val.png')"
            if task=='E4':code+="\npicture('tracking_comparison.png')"
        elif index==25:code="picture('synchronized_frame_0.png');picture('representative_detection_error.png');picture('tracking_timeline.png');picture('tracking_transition.png');print('Tracking transition views are qualitative matched-ID diagnostics, not official CLEAR IDSW event labels');display(pd.read_csv(RUN/'tables/error_examples.csv').groupby('method')[['tp','fp','fn']].sum())"
        elif index==26:code="coverage=json.loads((RUN/'coverage.json').read_text());display(pd.DataFrame([{'union_frames':coverage['decoded_union_frames'],'union_wall_seconds':coverage['union_wall_seconds']}]))\nprint('Processing seconds by method:',coverage['processing_seconds_by_method']);print('These scopes differ from historical warm YOLO processing FPS')"
        elif index==31:code="from soccertrack import config\nassert config.config_sha256()==data['config_sha256']\nprint('Current shared configuration matches accepted run')\nprint('Explicit expensive reproduction: see START_HERE.md and workflow.py stages; never automatic')"
        if code:cells.append(nbf.v4.new_code_cell(code))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'phase2','display_name':'Project phase2 venv','language':'python'}})
    target=ROOT/'notebooks/phase2_baseline.ipynb';nbf.write(nb,target)
    runtime=run/'notebook_runtime'
    for env,folder in [('IPYTHONDIR','ipython'),('JUPYTER_CONFIG_DIR','config'),('JUPYTER_DATA_DIR','data'),('JUPYTER_RUNTIME_DIR','runtime'),('TEMP','temp'),('TMP','temp')]:
        path=runtime/folder;path.mkdir(parents=True,exist_ok=True);os.environ[env]=str(path)
    kernel=runtime/'jupyter/kernels/phase2';kernel.mkdir(parents=True,exist_ok=True)
    write(kernel/'kernel.json',{'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Project venv','language':'python'})
    os.environ['JUPYTER_PATH']=str(runtime/'jupyter');os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    executed=NotebookClient(nb,timeout=180,kernel_name='phase2',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbf.write(executed,run/'phase2_baseline_executed.ipynb')
    exporter=HTMLExporter();exporter.embed_images=True;body,_=exporter.from_notebook_node(executed)
    (run/'phase2_baseline_executed.html').write_text(body,encoding='utf-8')
    code=[c for c in executed.cells if c.cell_type=='code'];errors=[o for c in code for o in c.get('outputs',[]) if o.output_type=='error']
    if errors or not all(c.execution_count is not None for c in code):raise RuntimeError('Notebook execution incomplete')
    write(run/'notebook_verification.json',dict(code_cells=len(code),executed_cells=len(code),error_outputs=len(errors),sections=len(sections),accepted_config_sha256=data['config_sha256']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--skip-visuals',action='store_true');a=p.parse_args();build(a.run.resolve(),not a.skip_visuals)
