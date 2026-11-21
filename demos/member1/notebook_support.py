"""Notebook context and explicit synthetic smoke execution (not real evidence)."""
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import cv2
import scientific_workflow as w
from experiment import demo_config, make_synthetic, collect_patches
from soccertrack import data_io,segmentation,proposals,tracker_iou

def make_smoke_run():
    run=w.BASE/'runs'/datetime.now(timezone.utc).strftime('synthetic_smoke_%Y%m%dT%H%M%S%fZ'); (run/'figures').mkdir(parents=True)
    cfg=demo_config(); cfg['cnn']['device']='cpu'; cfg['svm']['C_grid']=[.1]
    cfg['patches']['negative_mix']={'random':.7,'partial':.3,'proposal':0.}
    clip=run/'synthetic.avi'; gt=make_synthetic(clip,12); info=data_io.video_info(clip)
    arrays={}
    for name,frames in [('train',[0,2,4,6]),('val',[8,10])]:
        X,y,records=collect_patches(clip,gt,cfg,frames,42)
        rows=[{'patch':x,'label':int(label),'box':r[1:5],'kind':r[5],'frame':r[0],'match':'999001'} for x,label,r in zip(X,y,records)]
        arrays[name]=w.save_patch_arrays(rows,run/f'patches_{name}.npz','synthetic_'+name,cfg,seed_rule='master')
    models,metrics,meta=w.train_and_report(run,arrays['train'],arrays['val'],cfg,1)
    bg=segmentation.estimate_background(clip,5,12,cfg['segmentation']['work_scale'])
    gt.attrs['split']='train'; size=proposals.fit_size_model(gt); w.write_json(run/'size_model.json',size.__dict__)
    geometry=run/'local_geometry.npz'; np.savez_compressed(geometry,**{k:gt[k].to_numpy() for k in gt})
    cv2.imwrite(str(run/'background.png'),bg)
    quality={'match':'999001','split':'synthetic','before_rows':len(gt),'after_rows':len(gt),'issues':{},'raw_missing_values':0,'frames_annotated':12,'ids':3,'boxes_per_frame':{'min':3,'median':3,'max':3},'video_header':info,'cache_geometry':w.rel(geometry),'background':w.rel(run/'background.png')}
    data={'match':'999001','clip':clip,'gt':gt,'background':bg,'info':info,'quality':quality}
    summary,tables,stage,_=w.evaluate_frames(data,cfg,size,models,list(range(12)),run/'detection_baseline'); w.write_stage_images(run/'stages',stage,cfg,models)
    cfg['tracker_iou']['min_score']=.5; tracks=tracker_iou.track_sequence(tables['hog_svm'],range(12),cfg); tracks.to_csv(run/'tracking_local.csv',index=False)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(8,3))
    for tid,group in tracks.groupby('track_id'):
        if len(group)>1: plt.plot(group.x+group.w/2,group.y+group.h,label=str(tid))
    plt.gca().invert_yaxis(); plt.xlabel('Synthetic x pixels'); plt.ylabel('Foot-point y pixels')
    plt.title('SYNTHETIC predicted trajectories / correctness only'); w._figure(run/'figures/trajectories.png')
    plt.figure(figsize=(6,4))
    for name,table in tables.items():
        rows=[w.local_detection_diagnostics(table,gt,list(range(12)),float(t)) for t in np.arange(.1,1,.1)]
        plt.plot([r['recall'] for r in rows],[r['precision'] for r in rows],marker='o',label=name)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend(); plt.title('SYNTHETIC local detector thresholds')
    w._figure(run/'figures/detection_thresholds.png')
    image=next(data_io.iter_frames(clip,[2]))[1]
    from experiment import draw
    cv2.imwrite(str(run/'stages/10_tracking.png'),draw(image,tracks.loc[tracks.frame==2],(0,220,255),'SYNTHETIC ID'))
    w.dataset_figures(run,[data],arrays['train'],arrays['val'])
    manifest={'status':'SYNTHETIC SMOKE ONLY: same artificial clip, disjoint frames; not independent match evaluation','synthetic':True,'run_id':run.name,'models':meta,'dataset':[quality],'train_samples':len(arrays['train']['y']),'validation_samples':len(arrays['val']['y']),'classification_metrics':metrics,'paired_detection':{'baseline':summary},'recommended_configuration':'baseline','report_frames':list(range(12)),'seed':42,'frame_plan':{'synthetic_train':[0,2,4,6],'synthetic_val':[8,10]},'size_model':size.__dict__,'tracking':{'rows':len(tracks),'local_tracks':int(tracks.track_id.nunique()),'HOTA':'NOT RUN','IDF1':'NOT RUN'},'final_test':'NOT RUN','limitations':['Synthetic correctness demonstration only','Altered 0.7 random / 0.3 partial negative mix; no proposal negatives','No real generalization evidence']}
    w.write_json(run/'manifest.json',manifest); return run

def context(smoke=False):
    import torch
    torch.set_num_threads(2)
    run=make_smoke_run() if smoke else w.active_run(); m=w.read_json(run/'manifest.json'); cfg=w.read_json(run/'configuration.json')
    train=w.load_patch_arrays(run,'train'); val=w.load_patch_arrays(run,'val'); models=w.load_models(run)
    item=next((d for d in m['dataset'] if d['split']=='val'),m['dataset'][0]); gt=w.load_geometry(w.ROOT/item['cache_geometry']); bg=cv2.imread(str(w.ROOT/item['background']))
    if smoke: clip=run/'synthetic.avi'
    else: clip=w.safe_video(w.load_sources()['matches'][item['match']]['clip'])
    original=next(data_io.iter_frames(clip,[0]))[1]
    size=proposals.SizeModel(**m['size_model']); selected=m['recommended_configuration']; runtime=m.get('candidate_configuration',cfg) if selected=='candidate' else cfg
    supplement=None
    pointer=w.BASE/'corrections/latest.json'
    if not smoke and pointer.is_file():
        correction=w.read_json(pointer)
        if correction.get('source_run')==w.rel(run):supplement=w.read_json(w.ROOT/correction['path']/'historical_audit_supplement.json')
    return dict(run=run,m=m,cfg=cfg,train=train,val=val,models=models,item=item,gt=gt,bg=bg,clip=clip,original=original,size=size,runtime=runtime,supplement=supplement)
