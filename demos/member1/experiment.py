"""Isolated Member 1 PROVISIONAL experiment/demo; never an official shared runner.

Explicitly reads YAML and checks raw GT locally while M0 is unavailable. It does
not monkeypatch or replace shared configuration, contracts or metrics modules.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import time
import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score,precision_recall_fscore_support,roc_auc_score
from soccertrack import data_io,segmentation,proposals,patches,classifiers,detector_scratch,tracker_iou

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/member1_demo'
HELD_OUT={'128057','132831','117092'}


def safe_video(path):
    if str(path).startswith(('\\\\','//')): raise PermissionError('Local files only; UNC paths are not supported')
    path=Path(path).expanduser().resolve()
    if any(Path(part).stem in HELD_OUT for part in path.parts): raise PermissionError('Held-out test matches are prohibited in this demo')
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def demo_config():
    """Explicit provisional YAML reader, isolated from official shared config API."""
    cfg=yaml.safe_load((ROOT/'configs/scratch.yaml').read_text(encoding='utf-8'))
    cfg['project']=yaml.safe_load((ROOT/'configs/project.yaml').read_text(encoding='utf-8'))
    return cfg


def validate_experiment_pairs(train_video,train_mot,val_video,val_mot,cfg):
    """Check explicit arguments before reading any annotation content."""
    if Path(train_video).stem not in cfg['project']['split']['train'] or Path(val_video).stem not in cfg['project']['split']['val']:
        raise PermissionError('Real experiment needs approved disjoint train/validation matches')
    if Path(train_mot).stem!=Path(train_video).stem or Path(val_mot).stem!=Path(val_video).stem:
        raise ValueError('Video and MOT match IDs must agree within each split')


def raw_gt(path,size):
    """Raw-GT-only preview: reject dirty/out-of-frame data; no official validation claim."""
    path=safe_video(path); raw=data_io.read_mot_raw(path)
    if raw.duplicated().any() or raw.duplicated(['frame','id']).any(): raise ValueError('Provisional raw preview requires clean unique GT')
    W,H=size
    if ((raw.w<=1)|(raw.h<=1)|(raw.x<0)|(raw.y<0)|(raw.x+raw.w>W)|(raw.y+raw.h>H)).any():
        raise ValueError('Raw GT needs official cleaning; provisional preview refuses to repair it')
    return raw.rename(columns={'id':'track_id'}).assign(score=1.)[['frame','track_id','x','y','w','h','score']]


def make_synthetic(path,frames=30):
    path.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25.,(640,360))
    if not writer.isOpened(): raise RuntimeError('Cannot create synthetic video')
    rows=[]
    try:
        for f in range(frames):
            image=np.full((360,640,3),(40,140,40),dtype=np.uint8)
            for tid,(x,y,w,h) in enumerate([(80+3*f,150,20,45),(300+2*f,240,20,45),(500-3*f,80,18,40)],1):
                image[y:y+h,x:x+w]=(30,30,160)
                rows.append([f,tid,x,y,w,h,1.])
            writer.write(image)
    finally: writer.release()
    return pd.DataFrame(rows,columns=['frame','track_id','x','y','w','h','score'])


def collect_patches(video,gt,cfg,frame_ids,seed):
    """Bounded provisional experiment: random/partial negatives, proposal mix disabled.

    Production builder retains the approved three-category mixture. This isolated
    experiment records the different mixture and does not claim M4 integration.
    """
    settings=json.loads(json.dumps(cfg)); settings['patches']['negative_mix']={'random':.7,'partial':.3,'proposal':0.}
    rng=np.random.default_rng(seed); X=[]; y=[]; info=[]
    empty=pd.DataFrame(columns=['x','y','w','h'])
    for f,image in data_io.iter_frames(video,frame_ids):
        truth=gt.loc[gt.frame==f]
        records=patches.sample_frame_patches(image,truth,np.ones(image.shape[:2],bool),empty,settings,rng)
        for patch,label,box,kind in records:
            X.append(patch); y.append(label); info.append([f,*box,kind])
    if not X: raise ValueError('No experimental patches')
    return np.stack(X),np.asarray(y,dtype=np.int8),info


def binary_metrics(y,score):
    p,r,f,_=precision_recall_fscore_support(y,score>=.5,average='binary',zero_division=0)
    return {'precision_at_0_5':float(p),'recall_at_0_5':float(r),'f1_at_0_5':float(f),
            'pr_auc':float(average_precision_score(y,score)),'roc_auc':float(roc_auc_score(y,score)),
            'samples':len(y),'positive':int(y.sum())}


def local_detection_diagnostics(detections,gt,frames,threshold=.5,iou_threshold=.5):
    """Isolated provisional greedy counts; NOT Member 3's evaluator/Metric JSON."""
    if not 0<=threshold<=1 or not 0<iou_threshold<=1: raise ValueError('Invalid diagnostic threshold')
    if not detections.frame.isin(frames).all(): raise ValueError('Predictions outside diagnostic frame selection')
    tp=fp=0; total=0
    for frame in frames:
        truth=gt.loc[gt.frame==frame,['x','y','w','h']].to_numpy(float); total+=len(truth); used=set()
        pred=detections.loc[(detections.frame==frame)&(detections.score>=threshold)].sort_values('score',ascending=False)
        for box in pred[['x','y','w','h']].to_numpy(float):
            remaining=[j for j in range(len(truth)) if j not in used]
            if not remaining: fp+=1; continue
            overlap=tracker_iou._iou(np.array([box]),truth[remaining])[0]; best=int(overlap.argmax())
            if overlap[best]>=iou_threshold: tp+=1; used.add(remaining[best])
            else: fp+=1
    fn=total-tp; precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/total if total else 0.
    return {'status':'PROVISIONAL local greedy diagnostic; not official E3','score_threshold':threshold,'iou_threshold':iou_threshold,
            'frames':len(frames),'gt_boxes':total,'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}


def raw_summary(gt,video):
    counts=gt.groupby('frame').size(); lengths=gt.groupby('track_id').size()
    quant=lambda v:{k:float(v.quantile(q)) for k,q in [('p05',.05),('median',.5),('p95',.95)]}
    return {'frames_annotated':int(gt.frame.nunique()),'boxes':len(gt),'ids':int(gt.track_id.nunique()),
            'boxes_per_frame':{'min':int(counts.min()),'mean':float(counts.mean()),'max':int(counts.max())},
            'box_width_px':quant(gt.w),'box_height_px':quant(gt.h),
            'track_length_frames':{'min':int(lengths.min()),'median':float(lengths.median()),'max':int(lengths.max())},
            'video_header':data_io.video_info(video),'full_decoded_count':'NOT verified independently'}


def inspect_approved_labels(mot_root,cfg):
    """Bounded, explicitly provisional raw-label EDA for enabled train/val only."""
    from soccertrack import split as splitting
    if str(mot_root).startswith(('\\\\','//')): raise PermissionError('Local folders only')
    result={'status':'PROVISIONAL raw annotations; no shared cleaning or full video alignment verification','matches':{},'unavailable_enabled_matches':[]}
    samples=[]; distributions=[]
    for name in ('train','val'):
        for match in splitting.matches_for(name,cfg['project']):
            path=Path(mot_root)/f'{match}.txt'
            if not path.is_file():
                result['unavailable_enabled_matches'].append(match)
                continue
            raw=data_io.read_mot_raw(safe_video(path))
            if raw.empty: raise ValueError(f'Empty annotation file for {match}')
            import csv
            with path.open(newline='') as stream:
                decimal_tokens=sum(sum('.' in token for token in row[2:6]) for row in csv.reader(stream))
            counts=raw.groupby('frame').size(); lengths=raw.groupby('id').size()
            quant=lambda v:{key:float(v.quantile(q)) for key,q in [('p05',.05),('median',.5),('p95',.95)]}
            result['matches'][match]={'split':name,'frames_annotated':int(raw.frame.nunique()),'frame_min':int(raw.frame.min()),
                'frame_max':int(raw.frame.max()),'frame_gaps':int(raw.frame.max())+1-int(raw.frame.nunique()),'raw_boxes':len(raw),
                'ids':int(raw.id.nunique()),'boxes_per_frame':{'min':int(counts.min()),'max':int(counts.max()),'mean':float(counts.mean())},
                'box_width_px':quant(raw.w),'box_height_px':quant(raw.h),'track_length_frames':{'min':int(lengths.min()),'median':float(lengths.median()),'max':int(lengths.max())},
                'duplicate_rows':int(raw.duplicated().sum()),'degenerate_raw_boxes':int(((raw.w<=1)|(raw.h<=1)).sum()),
                'file_bytes':path.stat().st_size,'coordinate_tokens_with_decimal_point':decimal_tokens,
                'trailing_columns_all_minus_one':bool((raw[['c7','c8','c9','c10']]==-1).all().all()),
                'video_decoded_count':'NOT checked by raw-label inspection'}
            samples.append(raw.iloc[::7].assign(match=match,split=name)[['match','split','frame','x','y','w','h']])
            distributions.append((match,raw.w.to_numpy(),raw.h.to_numpy()))
    result['totals']={'matches':len(result['matches']),'annotated_frames':sum(m['frames_annotated'] for m in result['matches'].values()),'raw_boxes':sum(m['raw_boxes'] for m in result['matches'].values())}
    result['paper_reference_from_frozen_contract']={'frames':59842,'boxes':1316524,'median_box_wh_px':[25,55]}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'all_available_raw_eda_provisional.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if samples: pd.concat(samples,ignore_index=True).to_csv(OUT/'box_sample_provisional.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for match,width,height in distributions:
        for ax,values in zip(axes,[width,height]): ax.hist(values,bins=40,density=True,histtype='step',label=match)
    for ax,title in zip(axes,['width','height']): ax.set_title('PROVISIONAL raw '+title); ax.set_xlabel('Full-frame pixels'); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT/'eda_box_distributions_provisional.png'); plt.close(fig)
    return result


def save_demo_patchset(X,y,records,match,name,path,seed):
    """Write an explicitly provisional schema-shaped NPZ, not a validated export."""
    from soccertrack import contracts
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={'X':X,'y':y,'match':np.full(len(y),match,dtype='<U6'),
            'frame':np.array([r[0] for r in records],dtype=np.int32),
            'box':np.array([r[1:5] for r in records],dtype=np.float32),
            'kind':np.array([r[5] for r in records],dtype='<U8'),
            'meta_json':np.array([json.dumps({'schema_version':contracts.SCHEMA_VERSION,'split':name,'ratio':3,
                 'seed':seed,'sampling_stride':None,'iou_bounds':{'random_max':.1,'partial':[.1,.4]},
                 'status':'PROVISIONAL; shared validation unavailable','negative_mix':{'random':.7,'partial':.3,'proposal':0.}})])}
    np.savez_compressed(path,**arrays)
    return arrays


def train_models(X,y,Xv,yv,cfg,folder,epochs):
    folder.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(json.dumps(cfg)); cfg['cnn'].update(epochs=epochs,batch_size=64)
    cfg['svm']['C_grid']=[.01,.1,1.]
    svm=classifiers.HogSvmClassifier(cfg).fit_with_validation(X,y,Xv,yv)
    svm_path=folder/'hog_svm.joblib'; svm.save(svm_path)
    try:
        cnn=classifiers.CnnClassifier(cfg); history=cnn.fit(X,y,Xv,yv)
    except RuntimeError as exc:
        if cfg['cnn']['device']=='cpu': raise
        cfg['cnn']['device']='cpu'; cnn=classifiers.CnnClassifier(cfg); history=cnn.fit(X,y,Xv,yv)
    cnn_path=folder/'cnn.pt'; cnn.save(cnn_path)
    metrics={'hog_svm':binary_metrics(yv,svm.predict_proba(Xv)),'cnn':binary_metrics(yv,cnn.predict_proba(Xv)),
             'cnn_device':str(cnn.device),'cnn_epochs':len(history['epoch']),'cnn_parameters':sum(p.numel() for p in cnn.model.parameters()),
             'selected_C':svm.selected_C}
    (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
    (folder/'config.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(9,3))
    ax[0].plot(history['epoch'],history['train_loss'],label='train'); ax[0].plot(history['epoch'],history['val_loss'],label='val'); ax[0].legend(); ax[0].set_title('PROVISIONAL subset BCE')
    ax[1].plot(history['epoch'],history['val_pr_auc']); ax[1].set_title('PROVISIONAL validation PR-AUC')
    fig.tight_layout(); fig.savefig(folder/'learning_curves.png'); plt.close(fig)
    return svm,cnn,metrics,{'hog_svm':svm_path.relative_to(ROOT).as_posix(),'cnn':cnn_path.relative_to(ROOT).as_posix()}


def draw(image,table,color,label):
    image=image.copy()
    font_scale=.45*max(1,image.shape[1]/1280)
    for row in table.itertuples():
        x,y,w,h=map(lambda v:int(round(v)),(row.x,row.y,row.w,row.h))
        text=label
        if hasattr(row,'score'): text+=f' {row.score:.2f}'
        if hasattr(row,'track_id'): text+=f' ID {row.track_id}'
        cv2.rectangle(image,(x,y),(x+w,y+h),color,max(1,round(image.shape[1]/1280)))
        cv2.putText(image,text,(x,max(round(18*font_scale),y-4)),cv2.FONT_HERSHEY_SIMPLEX,font_scale,color,max(1,round(font_scale)))
    return image


def export_preview(video,gt,cfg,model,folder,classifier=None,limit=30,background_window=None,background=None,cancel_check=None,provenance=None):
    """Direct provisional demo outputs, never sent to the official output directories."""
    folder.mkdir(parents=True,exist_ok=True)
    info=data_io.video_info(video); count=min(limit,info['frame_count_header'])
    window=count if background_window is None else min(info['frame_count_header'],int(background_window))
    if not 1<=window<=6000: raise ValueError('Background window must be within 1..6000 frames')
    samples=min(9,window); background_start=time.perf_counter()
    bg=background if background is not None else segmentation.estimate_background(video,samples,window,cfg['segmentation']['work_scale'])
    background_seconds=time.perf_counter()-background_start
    pitch=segmentation.pitch_mask(bg,cfg,Path(video).stem)
    cv2.imwrite(str(folder/'background_provisional.png'),bg)
    cv2.imwrite(str(folder/'pitch_provisional.png'),pitch.astype(np.uint8)*255)
    options=cfg['tracker_iou']
    tracker=tracker_iou.IouHungarianTracker(options['iou_gate'],options['n_init'],options['max_age'],.5)
    path=folder/'annotated_provisional.mp4'; width=min(1280,info['width']); height=round(info['height']*width/info['width'])
    width-=width%2; height-=height%2
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),info['fps'],(width,height))
    if not writer.isOpened(): raise RuntimeError('Video export codec unavailable')
    all_det=[]; all_tracks=[]; recalls=[]; start=time.perf_counter()
    try:
        for f,image in data_io.iter_frames(video,range(count)):
            if cancel_check is not None: cancel_check()
            p,d,mask=detector_scratch.detect_frame(image,f,bg,pitch,model,cfg,classifier)
            tracks=tracker.update(f,d); all_det.append(d); all_tracks.append(tracks)
            if gt is not None:
                truth=gt.loc[gt.frame==f]
                if len(truth):
                    ov=tracker_iou._iou(truth[['x','y','w','h']].to_numpy(),p[['x','y','w','h']].to_numpy())
                    recalls.append(float((ov.max(axis=1)>=.5).mean()) if len(p) else 0.)
            annotated=draw(image,tracks,(0,200,255),'PROVISIONAL track')
            cv2.putText(annotated,'MEMBER 1 PROVISIONAL / shared validation unavailable',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            resized=cv2.resize(annotated,(width,height)); writer.write(resized)
            if f==0:
                cv2.imwrite(str(folder/'overlay_provisional.png'),resized)
                cv2.imwrite(str(folder/'foreground_provisional.png'),mask.astype(np.uint8)*255)
    finally: writer.release()
    seconds=time.perf_counter()-start
    if cancel_check is not None: cancel_check()
    pd.concat(all_det,ignore_index=True).to_csv(folder/'detections_provisional.csv',index=False)
    pd.concat(all_tracks,ignore_index=True).to_csv(folder/'tracks_provisional.csv',index=False)
    if provenance is not None:
        (folder/'provenance.json').write_text(json.dumps({'status':'PROVISIONAL export','provenance':provenance,'configuration':cfg,'frames':list(range(count))},indent=2)+'\n',encoding='utf-8')
    cap=cv2.VideoCapture(str(path)); decoded=0
    try:
        while cap.grab():
            ok,img=cap.retrieve()
            if not ok: raise RuntimeError('Export playback failed')
            decoded+=1
    finally: cap.release()
    if decoded!=count: raise RuntimeError('Exported video frame-count mismatch')
    diagnostics=local_detection_diagnostics(pd.concat(all_det,ignore_index=True),gt,range(count)) if gt is not None else None
    return {'frames':count,'export_frames_decoded':decoded,'seconds':seconds,'processing_including_overlay_export_fps':count/max(seconds,1e-9),
            'background_window_frames':window,'background_samples':samples,'background_seconds':background_seconds,
            'background_cache_hit':background is not None,
            'detection_diagnostic':diagnostics,
            'proposal_recall_at_0_5_local_geometry':float(np.mean(recalls)) if recalls else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-train-video'); parser.add_argument('--real-train-mot')
    parser.add_argument('--real-val-video'); parser.add_argument('--real-val-mot')
    parser.add_argument('--epochs',type=int,default=5); parser.add_argument('--frames',type=int,default=30)
    parser.add_argument('--background-window',type=int,default=None,help='optional label-free background window, maximum 6000 frames')
    parser.add_argument('--mot-root',type=Path,help='explicit local MOT folder for provisional enabled train/val raw-label EDA')
    parser.add_argument('--inspect-only',action='store_true',help='perform the raw-label inspection without training')
    args=parser.parse_args()
    if not 1<=args.epochs<=60 or not 1<=args.frames<=100: raise ValueError('Experiment budget exceeded')
    OUT.mkdir(parents=True,exist_ok=True); cfg=demo_config()
    if args.mot_root:
        inspection=inspect_approved_labels(args.mot_root,cfg)
        print(json.dumps(inspection,indent=2))
    if args.inspect_only:
        if not args.mot_root: raise ValueError('--inspect-only requires --mot-root')
        return
    synthetic=OUT/'synthetic.avi'; gt=make_synthetic(synthetic)
    X,y,meta=collect_patches(synthetic,gt,cfg,[0,5,10,15],42)
    Xv,yv,metav=collect_patches(synthetic,gt,cfg,[20,25],43)
    save_demo_patchset(X,y,meta,'999001','train',OUT/'synthetic_patchsets/patches_train.npz',42)
    save_demo_patchset(Xv,yv,metav,'999001','val',OUT/'synthetic_patchsets/patches_val.npz',43)
    # Synthetic train/val share a generated scene: diagnostic ONLY, never reported generalization.
    svm,cnn,metrics,checkpoints=train_models(X,y,Xv,yv,cfg,OUT/'synthetic_models',args.epochs)
    model=proposals.fit_size_model(gt.query('frame<20').copy())
    preview=export_preview(synthetic,gt,cfg,model,OUT/'synthetic_preview',svm,args.frames)
    report={'status':'PROVISIONAL; not shared Metrics JSON','synthetic':{'metrics':metrics,'checkpoints':checkpoints,'preview':preview,
             'size_model':model.__dict__,'limitations':'Synthetic train/validation frames share a scene; diagnostic only.'},'real':None,'dependencies':'Members 3 and 4 M0 unavailable; no shared interfaces patched'}
    if args.real_train_video:
        tv=safe_video(args.real_train_video); vv=safe_video(args.real_val_video)
        validate_experiment_pairs(tv,args.real_train_mot,vv,args.real_val_mot,cfg)
        ti=data_io.video_info(tv); vi=data_io.video_info(vv)
        train=raw_gt(args.real_train_mot,(ti['width'],ti['height'])); val=raw_gt(args.real_val_mot,(vi['width'],vi['height']))
        X,y,records=collect_patches(tv,train,cfg,[0,10,20,30],42); Xv,yv,records_v=collect_patches(vv,val,cfg,[0,10],43)
        save_demo_patchset(X,y,records,tv.stem,'train',OUT/'real_subset_patchsets/patches_train.npz',42)
        save_demo_patchset(Xv,yv,records_v,vv.stem,'val',OUT/'real_subset_patchsets/patches_val.npz',43)
        svm,cnn,metrics,checkpoints=train_models(X,y,Xv,yv,cfg,OUT/'real_subset_models',min(args.epochs,5))
        training=train.query('frame<=30').copy(); training.attrs['split']='train'; model=proposals.fit_size_model(training)
        preview=export_preview(vv,val,cfg,model,OUT/'real_val_preview',svm,min(args.frames,30),args.background_window)
        report['real']={'train_match':tv.stem,'validation_match':vv.stem,'train_samples':len(y),'val_samples':len(yv),
                        'train_frames':[0,10,20,30],'validation_frames':[0,10],'negative_mix':{'random':.7,'partial':.3,'proposal':0},
                        'metrics':metrics,'checkpoints':checkpoints,'preview':preview,'size_model':model.__dict__,'limitations':'Tiny provisional subset; no official E1/E3/E4, thresholds fixed at 0.5 for illustration.'}
        summary={'status':'PROVISIONAL raw GT inspection; not official stats.json',
                 'matches':{tv.stem:raw_summary(train,tv),vv.stem:raw_summary(val,vv)}}
        (OUT/'eda_provisional.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for group in ('synthetic','real'):
        if report[group]:
            report[group]['checkpoint_sha256']={k:hashlib.sha256((ROOT/v).read_bytes()).hexdigest() for k,v in report[group]['checkpoints'].items()}
    (OUT/'experiment_provisional.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
