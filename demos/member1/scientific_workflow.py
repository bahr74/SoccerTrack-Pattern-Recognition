"""Member 1 scientific workflow: explicit local adapters and versioned diagnostics.

No shared M0 function is patched or bypassed in the official pipeline. Local
geometry cleaning and all outputs here are marked uncertified/provisional.
"""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
import argparse, hashlib, json, time, sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,average_precision_score,roc_auc_score,confusion_matrix
from experiment import ROOT, OUT as LEGACY, demo_config, safe_video, draw, local_detection_diagnostics, make_synthetic, collect_patches
from soccertrack import data_io,segmentation,proposals,patches,features,classifiers,tracker_iou,detector_scratch
from provenance import indices,selection_metadata,resolved_plan,load_plan,patch_summary

BASE=ROOT/'outputs/member1_scientific'
STATUS='PROVISIONAL local validation; shared M0 certification unavailable; final test NOT RUN'
HEADINGS=['Problem Definition and Objectives','Dataset Description and Data Loading','Data Preparation','Data Cleaning',
'Exploratory Data Analysis (EDA)','Data Preprocessing','Segmentation and Proposal Generation','Baseline Model','Feature Extraction',
'Classification Models','Model Training','Validation, Testing and Evaluation','Detection Pipeline','Tracking Pipeline',
'Model Comparison and Error Analysis','Final Results, Conclusions and Reproducibility']


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def active_run():
    if not (BASE/'latest.json').is_file(): raise FileNotFoundError('No scientific run. Run scientific_workflow.py --run first, or select explicit smoke mode.')
    return ROOT/read_json(BASE/'latest.json')['path']


def register_sources(dataset_root):
    cfg=demo_config()['project']; root=Path(dataset_root).resolve(); matches={}
    for split in ('train','val'):
        for match in cfg['split'][split]:
            mot=root/f'data/external/soccertrack-v2/mot/{match}.txt'
            clips=[root/f'data/external/soccertrack-v2/mot/clips/{match}.mp4',root/f'data/external/soccertrack-v2/raw/{match}/{match}.mp4']
            clip=next((p for p in clips if p.is_file()),None)
            if not mot.is_file() or clip is None: raise FileNotFoundError(f'Approved {split} match {match} is incomplete')
            matches[match]={'split':split,'mot':str(safe_video(mot)),'clip':str(safe_video(clip))}
    value={'status':'Isolated Member 1 local registry, not shared config.load_data_sources','matches':matches}
    write_json(BASE/'local_sources.json',value); return value


def load_sources(): return read_json(BASE/'local_sources.json')


def load_geometry(path):
    with np.load(path,allow_pickle=False) as archive: return pd.DataFrame({name:archive[name] for name in archive.files})


def source_data(match,sources,cfg,background=True):
    item=sources['matches'][match]
    if item['split'] not in ('train','val') or match not in cfg['project']['split'][item['split']]: raise PermissionError('Unapproved match/split')
    clip=safe_video(item['clip']); mot=safe_video(item['mot']); info=data_io.video_info(clip)
    if clip.stem!=match or mot.stem!=match: raise ValueError('Registered video and MOT identities must agree with the match ID')
    sample_ids=indices(info['frame_count_header'],cfg['sampling']['background_num_frames'])
    signature=hashlib.sha256(('resolved_v2'+str(clip)+str(clip.stat().st_size)+str(clip.stat().st_mtime_ns)+sha(mot)+str(cfg['segmentation']['work_scale'])+str(sample_ids)).encode()).hexdigest()[:18]
    folder=BASE/'cache'/f'{match}_{signature}'; folder.mkdir(parents=True,exist_ok=True)
    geometry=folder/'local_geometry.npz'; stats_path=folder/'data_quality.json'
    if geometry.is_file() and stats_path.is_file(): gt=load_geometry(geometry); stats=read_json(stats_path)
    else:
        raw=data_io.read_mot_raw(mot)
        gt,issues=data_io.clean_mot_geometry(raw,(info['width'],info['height']),22)
        np.savez_compressed(geometry,**{name:gt[name].to_numpy() for name in gt.columns})
        counts=gt.groupby('frame').size(); lengths=gt.groupby('track_id').size()
        stat=lambda s:{'min':float(s.min()),'median':float(s.median()),'max':float(s.max())}
        stats={'match':match,'split':item['split'],'before_rows':len(raw),'after_rows':len(gt),'issues':issues,
               'raw_missing_values':int(raw.isna().sum().sum()),'frames_annotated':int(raw.frame.nunique()),'frame_min':int(raw.frame.min()),'frame_max':int(raw.frame.max()),
               'ids':int(gt.track_id.nunique()),'boxes_per_frame':stat(counts),'track_length_frames':stat(lengths),
               'box_width_px':stat(gt.w),'box_height_px':stat(gt.h),'box_area_px':stat(gt.w*gt.h),
               'video_header':info,'alignment':'0-based sequential sample alignment; full decoded-count audit NOT RUN',
               'official_validation':'NOT RUN: shared validator unavailable','annotation_sha256':sha(mot),'source_fingerprint':'Video path/size/mtime, annotation SHA-256; not a full video hash'}
        write_json(stats_path,stats)
    bg_path=folder/'background.png'; started=time.perf_counter()
    if background and not bg_path.is_file():
        print(f'Background {match}: {len(sample_ids)} samples over {info["frame_count_header"]} sequential indices',flush=True)
        bg=segmentation.estimate_background(clip,cfg['sampling']['background_num_frames'],info['frame_count_header'],cfg['segmentation']['work_scale'])
        if not cv2.imwrite(str(bg_path),bg): raise OSError('Background cache write failed')
    bg=cv2.imread(str(bg_path)) if background else None
    return {'match':match,'clip':clip,'mot':mot,'info':info,'gt':gt,'quality':stats,'cache':folder,'geometry':geometry,'background':bg,'background_seconds_this_call':time.perf_counter()-started}


def save_patch_arrays(records,path,name,cfg,seed_rule='master_plus_match_id'):
    if not records: raise ValueError('No patches available')
    X=np.stack([r['patch'] for r in records]); y=np.array([r['label'] for r in records],np.int8)
    arrays={'X':X,'y':y,'match':np.array([r['match'] for r in records],dtype='<U6'),
            'frame':np.array([r['frame'] for r in records],np.int32),'box':np.array([r['box'] for r in records],np.float32),
            'kind':np.array([r['kind'] for r in records],dtype='<U8'),
            'meta_json':np.array([json.dumps({'schema_version':'1.0','split':name,'ratio':cfg['patches']['negatives_per_positive'],
                'seed':cfg['project']['seed'],**selection_metadata(records,cfg,seed_rule),'iou_bounds':{'random_max':cfg['patches']['random_iou_max'],'partial':cfg['patches']['partial_iou_range']},
                'negative_mix':cfg['patches']['negative_mix'],'status':STATUS})])}
    if X.shape[1:]!=(64,32,3) or set(np.unique(y))!={0,1}: raise ValueError('Invalid local patch arrays')
    if int((y==0).sum())!=3*int((y==1).sum()): raise ValueError('Configured 3:1 balance not achieved')
    if name in ('train','val') and not set(arrays['match']).issubset(cfg['project']['split'][name]): raise PermissionError('Patch leakage guard')
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**arrays)
    return arrays


def load_patch_arrays(run,name):
    with np.load(Path(run)/f'patches_{name}.npz',allow_pickle=False) as archive: return {k:archive[k] for k in archive.files}


def collect_dataset(data,cfg,size_model,frames):
    gt=data['gt']; bg=data['background']; info=data['info']; records=[]; skipped=[]
    pitch=segmentation.pitch_mask(bg,cfg,data['match'])
    fullpitch=cv2.resize(pitch.astype(np.uint8),(info['width'],info['height']),interpolation=cv2.INTER_NEAREST).astype(bool)
    rng=np.random.default_rng(cfg['project']['seed']+int(data['match']))
    for index,image in data_io.iter_frames(data['clip'],frames):
        candidates,_,_=detector_scratch.prepare_frame_proposals(image,index,bg,pitch,size_model,cfg)
        try:
            rows=patches.sample_frame_patches(image,gt.loc[gt.frame==index],fullpitch,candidates,cfg,rng)
        except ValueError as exc:
            skipped.append({'match':data['match'],'frame':index,'reason':str(exc)}); continue
        for patch,label,box,kind in rows:
            records.append({'patch':patch,'label':label,'box':box,'kind':kind,'frame':index,'match':data['match']})
    print(f'Patches {data["match"]}: {len(records)} rows, {len(skipped)} excluded frames',flush=True)
    return records,skipped


def patch_metrics(y,score):
    p,r,f,_=precision_recall_fscore_support(y,score>=.5,average='binary',zero_division=0)
    return {'accuracy':float(accuracy_score(y,score>=.5)),'precision':float(p),'recall':float(r),'f1':float(f),
            'pr_auc_average_precision':float(average_precision_score(y,score)),'roc_auc':float(roc_auc_score(y,score)),
            'confusion_matrix':confusion_matrix(y,score>=.5,labels=[0,1]).tolist(),'threshold':.5}


def load_models(run,device='cpu'):
    run=Path(run); manifest=read_json(run/'manifest.json'); models={}
    for name,cls in [('hog_svm',classifiers.HogSvmClassifier),('cnn',classifiers.CnnClassifier)]:
        path=ROOT/manifest['models'][name]['checkpoint']
        if sha(path)!=manifest['models'][name]['sha256']: raise ValueError('Checkpoint hash mismatch')
        models[name]=cls.load(path)
    if device=='cuda':
        import torch
        if torch.cuda.is_available(): models['cnn'].device=torch.device('cuda'); models['cnn'].model.to('cuda')
    return models


def _figure(path):
    plt.tight_layout(); plt.savefig(path,dpi=130,bbox_inches='tight'); plt.close()


def plot_classification(run,arrays,metrics,scores,history):
    fig,axes=plt.subplots(1,3,figsize=(11,3))
    for ax,(name,value) in zip(axes,metrics.items()):
        ax.imshow(value['confusion_matrix'],cmap='Blues'); ax.set_title(name+' / validation patches'); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        for (y,x),v in np.ndenumerate(np.array(value['confusion_matrix'])): ax.text(x,y,str(v),ha='center',va='center')
    _figure(run/'figures/confusion_matrices.png')
    from sklearn.metrics import precision_recall_curve
    plt.figure(figsize=(6,4))
    for name,score in scores.items():
        precision,recall,_=precision_recall_curve(arrays['y'],score); plt.plot(recall,precision,label=name)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Validation PATCH classification (not detection)'); plt.legend(); _figure(run/'figures/patch_pr_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].plot(history['epoch'],history['train_loss'],label='train'); axes[0].plot(history['epoch'],history['val_loss'],label='validation'); axes[0].legend(); axes[0].set_title('Weighted BCE / augmented training')
    axes[1].plot(history['epoch'],history['val_pr_auc']); axes[1].set_title('Validation patch AP'); axes[1].set_xlabel('Epoch')
    _figure(run/'figures/training_curves.png')


def dataset_figures(run,data,train,val):
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    for item in data:
        gt=item['gt']; axes[0].hist(gt.w,bins=30,histtype='step',density=True,label=item['match']); axes[1].hist(gt.h,bins=30,histtype='step',density=True)
        axes[2].hist(gt.w*gt.h,bins=30,histtype='step',density=True)
    for ax,title in zip(axes,['Box width','Box height','Box area']): ax.set_title(title+' / locally cleaned GT'); ax.set_xlabel('Pixels' if title!='Box area' else 'Pixels squared')
    axes[0].legend(fontsize=6); _figure(run/'figures/eda_boxes.png')
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for item in data:
        gt=item['gt']; sampled=gt.iloc[::50]; axes[0].scatter(sampled.x+sampled.w/2,sampled.y+sampled.h,s=1,alpha=.15,label=item['match'])
        lengths=gt.groupby('track_id').size(); axes[1].hist(lengths,bins=15,histtype='step',label=item['match'])
    axes[0].invert_yaxis(); axes[0].set_title('Player foot-point spatial distribution'); axes[1].set_title('Track lengths / annotated frames'); _figure(run/'figures/eda_spatial_tracks.png')
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,array,name in zip(axes,[train,val],['train','validation']):
        unique,count=np.unique(array['kind'],return_counts=True); ax.bar(unique,count); ax.set_title(name+' sampling categories')
    _figure(run/'figures/class_balance.png')
    subset=train['X'][:12]; canvas=np.zeros((64*2,32*6,3),np.uint8)
    for i,patch in enumerate(subset): canvas[(i//6)*64:(i//6+1)*64,(i%6)*32:(i%6+1)*32]=patch
    plt.figure(figsize=(8,3)); plt.imshow(canvas); plt.axis('off'); plt.title('Actual train RGB patches / source metadata saved'); _figure(run/'figures/train_patches.png')
    descriptor=features.hog_features(train['X'][:1],demo_config())[0]
    fig,axes=plt.subplots(1,2,figsize=(5,4)); axes[0].imshow(train['X'][0]); axes[0].set_title('Train RGB patch'); axes[1].imshow(features.hog_glyph(descriptor,demo_config()),cmap='magma'); axes[1].set_title('756-D HOG glyph')
    _figure(run/'figures/hog_example.png')


def match_frame(detections,truth,threshold=.5,iou=.5):
    selected=detections.loc[detections.score>=threshold].sort_values('score',ascending=False)
    used=set(); matched=[]; fp=[]
    for index,row in selected.iterrows():
        available=[i for i in truth.index if i not in used]
        if not available: fp.append(index); continue
        overlap=tracker_iou._iou(np.array([[row.x,row.y,row.w,row.h]]),truth.loc[available,['x','y','w','h']].to_numpy())[0]
        best=int(overlap.argmax())
        if overlap[best]>=iou: used.add(available[best]); matched.append((index,available[best]))
        else: fp.append(index)
    return matched,fp,[i for i in truth.index if i not in used]


def evaluate_frames(data,cfg,size_model,models,frames,save_folder=None):
    bg=data['background']; pitch=segmentation.pitch_mask(bg,cfg,data['match']); tables={name:[] for name in ['rules','hog_svm','cnn']}
    proposal_hits=0; total=0; stage0=None; records=[]; clocks={name:0. for name in tables}; errors={name:[] for name in tables}
    for index,image in data_io.iter_frames(data['clip'],frames):
        started=time.perf_counter(); p,raw,clean=detector_scratch.prepare_frame_proposals(image,index,bg,pitch,size_model,cfg); prep=time.perf_counter()-started
        truth=data['gt'].loc[data['gt'].frame==index]; total+=len(truth)
        overlap=tracker_iou._iou(truth[['x','y','w','h']].to_numpy(),p[['x','y','w','h']].to_numpy())
        proposal_hits+=int((overlap.max(axis=1)>=.5).sum()) if len(p) else 0
        n,_,stats,_=cv2.connectedComponentsWithStats(clean.astype(np.uint8))
        rawboxes=np.array([[x,y,w,h] for x,y,w,h,_ in stats[1:]],dtype=float).reshape(-1,4)/cfg['segmentation']['work_scale']
        for name in tables:
            start=time.perf_counter(); scored,det=detector_scratch.classify_candidates(image,p,cfg,models.get(name)); clocks[name]+=time.perf_counter()-start+prep
            tables[name].append(det); matched,fp,fn=match_frame(det,truth)
            for truth_idx in truth.index:
                row=truth.loc[truth_idx]; box=np.array([[row.x,row.y,row.w,row.h]])
                if truth_idx in {j for _,j in matched}: category='detected'
                elif len(scored) and (tracker_iou._iou(box,scored.loc[scored.score>=.5,['x','y','w','h']].to_numpy())>=.5).any(): category='NMS_or_one_to_one_overlap'
                elif len(p) and (tracker_iou._iou(box,p[['x','y','w','h']].to_numpy())>=.5).any(): category='classification_or_threshold'
                elif len(rawboxes) and (tracker_iou._iou(box,rawboxes)>=.5).any(): category='proposal_filter_or_split'
                else: category='foreground_or_localization'
                errors[name].append({'frame':index,'track_id':int(row.track_id),'category':category})
            if name=='hog_svm': records.append({'frame':index,'fp':len(fp),'fn':len(fn),'tp':len(matched)})
            if index==frames[0] and name=='hog_svm': stage0={'frame':image.copy(),'proposals':p.copy(),'scored':scored.copy(),'detections':det.copy(),'raw':raw.copy(),'clean':clean.copy(),'pitch':pitch.copy(),'background':bg.copy(),'truth':truth.copy()}
    combined={name:pd.concat(rows,ignore_index=True) for name,rows in tables.items()}
    metrics={name:local_detection_diagnostics(table,data['gt'],frames) for name,table in combined.items()}
    for name in metrics: metrics[name]['processing_fps']=len(frames)/max(clocks[name],1e-9); metrics[name]['fps_excludes']='Decode, background preparation, file writing and overlay drawing'
    summary={'frame_ids':list(frames),'proposal_recall_at_0_5':proposal_hits/total if total else None,'gt_boxes':total,'models':metrics,
             'error_counts':{name:pd.Series([r['category'] for r in rows]).value_counts().to_dict() for name,rows in errors.items()}}
    if save_folder:
        save_folder.mkdir(parents=True,exist_ok=True)
        for name,table in combined.items(): table.to_csv(save_folder/f'{name}_detections_local.csv',index=False)
        write_json(save_folder/'summary.json',summary); write_json(save_folder/'errors.json',errors)
    return summary,combined,stage0,records


def write_stage_images(folder,stage,cfg,models):
    folder.mkdir(parents=True,exist_ok=True); frame=stage['frame']; width=min(1280,frame.shape[1]); height=round(frame.shape[0]*width/frame.shape[1])
    def save(name,image):
        if image.dtype==bool: image=image.astype(np.uint8)*255
        if not cv2.imwrite(str(folder/name),cv2.resize(image,(width,height),interpolation=cv2.INTER_NEAREST if image.ndim==2 else cv2.INTER_AREA)): raise OSError('Figure write failed')
    save('01_original.png',frame); save('02_background.png',stage['background']); save('03_pitch.png',stage['pitch']); save('04_foreground.png',stage['raw']); save('05_morphology.png',stage['clean'])
    save('06_proposals.png',draw(frame,stage['proposals'],(255,160,0),'Candidate'))
    X=np.stack([patches.crop_patch(frame,tuple(box)) for box in stage['proposals'][['x','y','w','h']].to_numpy()]) if len(stage['proposals']) else np.zeros((1,64,32,3),np.uint8)
    canvas=np.zeros((128,32*min(8,len(X)),3),np.uint8)
    for i,patch in enumerate(X[:8]): canvas[:64,i*32:(i+1)*32]=cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
    # A contact sheet has its own aspect ratio, not the panorama's aspect ratio.
    if not cv2.imwrite(str(folder/'07_patches.png'),canvas): raise OSError('Patch contact-sheet write failed')
    image=frame.copy()
    for method,color in [('hog_svm',(0,220,255)),('cnn',(255,130,120))]:
        _,d=detector_scratch.classify_candidates(frame,stage['proposals'],cfg,models[method]); image=draw(image,d.loc[d.score>=.5],color,method)
    save('08_classification.png',image)
    save('09_detections.png',draw(frame,stage['detections'].loc[stage['detections'].score>=.5],(0,220,255),'HOG/SVM'))
    save('gt_overlay.png',draw(frame,stage['truth'],(0,255,0),'Ground Truth'))
    matches,fp,fn=match_frame(stage['detections'],stage['truth'])
    errors=draw(frame,stage['truth'].loc[fn],(20,20,255),'FN / missed GT')
    errors=draw(errors,stage['detections'].loc[fp],(255,0,255),'FP / unmatched')
    save('failure_example.png',errors)


def train_and_report(run,train,val,cfg,epochs):
    import torch
    torch.set_num_threads(2)
    cfg=deepcopy(cfg); cfg['cnn']['epochs']=epochs; cfg['cnn']['batch_size']=128
    dummy=DummyClassifier(strategy='most_frequent').fit(np.zeros((len(train['y']),1)),train['y'])
    dummy_scores=dummy.predict_proba(np.zeros((len(val['y']),1)))[:,1]
    print('Training full-grid calibrated HOG/SVM',flush=True)
    start=time.perf_counter(); svm=classifiers.HogSvmClassifier(cfg).fit_with_validation(train['X'],train['y'],val['X'],val['y']); svm_seconds=time.perf_counter()-start
    print('Training bounded CNN',flush=True)
    start=time.perf_counter(); cnn=classifiers.CnnClassifier(cfg); history=cnn.fit(train['X'],train['y'],val['X'],val['y']); cnn_seconds=time.perf_counter()-start
    folder=run/'models'; folder.mkdir(); svm.save(folder/'hog_svm.joblib'); cnn.save(folder/'cnn.pt')
    scores={'majority_baseline':dummy_scores,'hog_svm':svm.predict_proba(val['X']),'cnn':cnn.predict_proba(val['X'])}
    metrics={name:patch_metrics(val['y'],score) for name,score in scores.items()}
    write_json(run/'classification_metrics.json',metrics); write_json(run/'training_history.json',history); write_json(run/'svm_search.json',svm.search_results_); write_json(run/'configuration.json',cfg)
    for name,score in scores.items():
        pd.DataFrame({'sample_id':range(len(score)),'match':val['match'],'frame':val['frame'],'kind':val['kind'],'y_true':val['y'],'score':score}).to_csv(run/f'validation_patch_scores_{name}.csv',index=False)
    plot_classification(run,val,metrics,scores,history)
    return {'hog_svm':svm,'cnn':cnn},metrics,{'hog_svm':{'checkpoint':rel(folder/'hog_svm.joblib'),'sha256':sha(folder/'hog_svm.joblib'),'selected_C':svm.selected_C,'training_seconds':svm_seconds},
           'cnn':{'checkpoint':rel(folder/'cnn.pt'),'sha256':sha(folder/'cnn.pt'),'epochs':len(history['epoch']),'parameters':sum(p.numel() for p in cnn.model.parameters()),'device':str(cnn.device),'training_seconds':cnn_seconds}}


def run_experiment(epochs=8,replay_settings=None,prepare_only=False):
    if not 1<=epochs<=15: raise ValueError('Training budget is 1..15 epochs')
    replay=load_plan(replay_settings) if replay_settings else None
    sources=load_sources(); cfg=deepcopy(replay['configuration']) if replay else demo_config()
    if replay:
        epochs=int(cfg['cnn']['epochs'])
    else:
        cfg['sampling']['background_num_frames']=9
        cfg['cnn'].update(device='auto',epochs=epochs,batch_size=128)
    if not 1<=epochs<=15: raise ValueError('Saved training budget must be within 1..15 epochs')
    run=BASE/'runs'/datetime.now(timezone.utc).strftime('expanded_%Y%m%dT%H%M%S%fZ'); (run/'figures').mkdir(parents=True)
    write_json(run/'progress.json',{'stage':'dataset preparation'})
    prepared=[]
    for match in cfg['project']['split']['train']+cfg['project']['split']['val']:
        if replay and match not in replay['frame_plan']: continue
        print('Preparing '+match,flush=True); prepared.append(source_data(match,sources,cfg))
    training=pd.concat([d['gt'] for d in prepared if d['quality']['split']=='train'],ignore_index=True); training.attrs['split']='train'
    size_model=proposals.fit_size_model(training); write_json(run/'size_model.json',size_model.__dict__)
    train_records=[]; val_records=[]; skipped=[]; frame_plan={}
    for data in prepared:
        total=data['info']['frame_count_header']; name=data['quality']['split']
        frames=replay['frame_plan'][data['match']] if replay else ((np.linspace(0,min(total-1,5250),8,dtype=int)//10*10).tolist() if name=='train' else [100,500,900,1300,1700,2100,2500,2900])
        if replay:
            bg_spec=replay['backgrounds'][data['match']]
            if total!=bg_spec['total_frames']: raise ValueError('Replay source frame count differs from recorded settings')
            expected=replay['inputs'].get(data['match'],{}).get('annotation_sha256')
            if expected and data['quality']['annotation_sha256']!=expected: raise ValueError('Replay annotation hash mismatch')
            expected_bg=replay['inputs'].get(data['match'],{}).get('background_sha256')
            if expected_bg and sha(data['cache']/'background.png')!=expected_bg: raise ValueError('Replay background hash mismatch')
        frame_plan[data['match']]=frames
        rows,excluded=collect_dataset(data,cfg,size_model,frames); skipped.extend(excluded)
        (train_records if name=='train' else val_records).extend(rows)
    train=save_patch_arrays(train_records,run/'patches_train.npz','train',cfg); val=save_patch_arrays(val_records,run/'patches_val.npz','val',cfg)
    if set(train['match']) & set(val['match']): raise PermissionError('Train/validation match overlap')
    write_json(run/'excluded_frames.json',skipped)
    backgrounds={d['match']:{'total_frames':d['info']['frame_count_header'],'sample_indices':indices(d['info']['frame_count_header'],cfg['sampling']['background_num_frames']),'effective_count':len(indices(d['info']['frame_count_header'],cfg['sampling']['background_num_frames']))} for d in prepared}
    inputs={d['match']:{'annotation_sha256':d['quality']['annotation_sha256'],'background_sha256':sha(d['cache']/'background.png'),'video_fingerprint_policy':'path/size/mtime cache identity; not a full source-video hash'} for d in prepared}
    settings=resolved_plan(cfg,frame_plan,backgrounds,inputs);write_json(run/'resolved_settings.json',settings)
    write_json(run/'duplication_summary.json',{name:patch_summary(arrays) for name,arrays in [('train',train),('val',val)]})
    if prepare_only:
        write_json(run/'configuration.json',cfg);write_json(run/'progress.json',{'stage':'prepared only; no model training'});print('PREPARED '+str(run),flush=True);return run
    write_json(run/'progress.json',{'stage':'model training','train_patches':len(train['y']),'val_patches':len(val['y'])})
    models,metrics,model_meta=train_and_report(run,train,val,cfg,epochs)
    dataset_figures(run,prepared,train,val)
    data=next(d for d in prepared if d['quality']['split']=='val')
    baseline=deepcopy(cfg); candidate=deepcopy(cfg)
    candidate['segmentation'].update(diff_threshold=18,open_ksize=1,close_ksize=5)
    candidate['proposals'].update(min_area_px=40,size_tolerance=[.35,2.])
    dev_frames=[200,250,300]; audit_frames=list(range(20))
    print('Paired segmentation development comparison',flush=True)
    dev_a,_,_,_=evaluate_frames(data,baseline,size_model,models,dev_frames)
    dev_b,_,_,_=evaluate_frames(data,candidate,size_model,models,dev_frames)
    use_candidate=dev_b['models']['hog_svm']['f1']>=dev_a['models']['hog_svm']['f1']
    chosen='candidate' if use_candidate else 'baseline'
    print('Paired diagnostic on identical frames 0..19',flush=True)
    a,ta,stage_a,_=evaluate_frames(data,baseline,size_model,models,audit_frames,run/'detection_baseline')
    b,tb,stage_b,_=evaluate_frames(data,candidate,size_model,models,audit_frames,run/'detection_candidate')
    selected=tb if use_candidate else ta; selected_stage=stage_b if use_candidate else stage_a; selected_cfg=candidate if use_candidate else baseline
    write_stage_images(run/'stages',selected_stage,selected_cfg,models)
    curves=[]
    for name,table in selected.items():
        for threshold in np.arange(.1,1,.1): curves.append(dict(local_detection_diagnostics(table,data['gt'],audit_frames,float(threshold)),method=name))
    pd.DataFrame(curves).to_csv(run/'detection_threshold_curves.csv',index=False)
    plt.figure(figsize=(7,4))
    for name in selected:
        rows=[r for r in curves if r['method']==name]; plt.plot([r['recall'] for r in rows],[r['precision'] for r in rows],marker='o',label=name)
    plt.xlabel('Detection recall'); plt.ylabel('Detection precision'); plt.title('Paired validation DETECTION thresholds / local diagnostics'); plt.legend(); _figure(run/'figures/detection_thresholds.png')
    settings=deepcopy(selected_cfg); settings['tracker_iou']['min_score']=.5
    tracks=tracker_iou.track_sequence(selected['hog_svm'],range(20),settings); tracks.to_csv(run/'tracking_local.csv',index=False)
    path=run/'annotated_validation.mp4'; width=1280; height=336
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),25.,(width,height)); assert writer.isOpened()
    try:
        for index,image in data_io.iter_frames(data['clip'],range(20)):
            annotated=draw(image,tracks.loc[tracks.frame==index],(0,200,255),'HOG/SVM ID')
            cv2.putText(annotated,'LOCAL VALIDATION DIAGNOSTIC - NOT OFFICIAL FINAL TEST',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),2)
            writer.write(cv2.resize(annotated,(width,height)))
            if index==2: cv2.imwrite(str(run/'stages/10_tracking.png'),cv2.resize(annotated,(width,height)))
    finally: writer.release()
    plt.figure(figsize=(9,3))
    for tid,group in tracks.groupby('track_id'):
        if len(group)>1: plt.plot(group.x+group.w/2,group.y+group.h,label=str(tid))
    plt.gca().invert_yaxis(); plt.xlabel('Full-frame x pixels'); plt.ylabel('Foot-point y pixels'); plt.title('Predicted trajectories / first 20 validation frames (local IDs)'); _figure(run/'figures/trajectories.png')
    match_meta=[dict(d['quality'],cache_geometry=rel(d['geometry']),background=rel(d['cache']/'background.png')) for d in prepared]
    manifest={'status':STATUS,'created_utc':datetime.now(timezone.utc).isoformat(),'run_id':run.name,'synthetic':False,'models':model_meta,
        'dataset':match_meta,'frame_plan':frame_plan,'train_samples':len(train['y']),'validation_samples':len(val['y']),
        'train_positive':int(train['y'].sum()),'validation_positive':int(val['y'].sum()),
        'training_matches':sorted(set(train['match'])),'validation_matches':sorted(set(val['match'])),
        'negative_mix':cfg['patches']['negative_mix'],'excluded_frames':skipped,'seed':42,'classification_metrics':metrics,
        'development_frames':dev_frames,'report_frames':audit_frames,'development':{'baseline':dev_a,'candidate':dev_b},
        'paired_detection':{'baseline':a,'candidate':b},'recommended_configuration':chosen,
        'tracking':{'local_tracks':int(tracks.track_id.nunique()),'rows':len(tracks),'HOTA':'NOT RUN','IDF1':'NOT RUN','reason':'Shared approved evaluator unavailable'},
        'size_model':size_model.__dict__,'candidate_configuration':candidate,'resolved_settings_sha256':settings['settings_sha256'],
        'artifact_sha256':{rel(p):sha(p) for p in sorted(run.rglob('*')) if p.is_file()},'final_test':'NOT RUN / no held-out access',
        'limitations':['Small correlated validation sample, not whole-match evaluation','Model selection uses validation patches; patch metrics are development estimates','M0 certification unavailable','Width/rule context remains provisional','OpenCV HOG reference unavailable']}
    write_json(run/'manifest.json',manifest); write_json(run/'progress.json',{'stage':'complete'}); write_json(BASE/'latest.json',{'path':rel(run)})
    print('COMPLETE '+str(run),flush=True); return run


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--register-data',type=Path); parser.add_argument('--run',action='store_true'); parser.add_argument('--epochs',type=int,default=8)
    parser.add_argument('--replay-settings',type=Path);parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.register_data: print(json.dumps(register_sources(args.register_data),indent=2))
    if args.replay_settings and not (args.run or args.prepare_only):
        plan=load_plan(args.replay_settings);print(json.dumps({'status':'Validated only; no data preparation or training','settings_sha256':plan['settings_sha256'],'frame_plan':plan['frame_plan']},indent=2))
    if args.run or args.prepare_only: run_experiment(args.epochs,args.replay_settings,args.prepare_only)

if __name__=='__main__': main()
