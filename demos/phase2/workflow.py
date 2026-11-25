"""Resumable integrated Phase2 validation workflow. No training or protected-data access."""
from pathlib import Path
from datetime import datetime,timezone
from copy import deepcopy
import argparse,hashlib,json,os,shutil,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import cv2
import yaml
from soccertrack import config,contracts as C,data_io,split,features,classifiers,detector_scratch as D,proposals,segmentation,evaluate as E,compare,tracker_iou,tracker_bytetrack

M1=ROOT/'outputs/member1_scientific/runs/expanded_20261120T044321992233Z'
M2=ROOT/'outputs/member2_scientific/runs/member2_validation_20261125'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def copy_new(source,target):
    target=Path(target)
    if target.exists():
        if sha(source)!=sha(target):raise FileExistsError(target)
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
def journal(run,message):
    with (run/'EXECUTION_JOURNAL.md').open('a',encoding='utf-8') as f:f.write('\n'+datetime.now(timezone.utc).isoformat()+' — '+message+'\n')
def mark(run,stage):
    state=read(run/'stage.json') if (run/'stage.json').exists() else {}
    state[stage]={'completed_utc':datetime.now(timezone.utc).isoformat(),'config_sha256':config.config_sha256()}
    write(run/'stage.json',state);journal(run,stage+' completed')
def cfg():return config.load_config()

def publish(run):
    m=read(M1/'manifest.json'); local=read(ROOT/'outputs/member1_scientific/local_sources.json')
    registry={'matches':{mid:{k:v for k,v in local['matches'][mid].items() if k in ('mot','clip','calibration_dir')}
                         for mid in ['118575','118576','118577','118578','128058','117093']},
              'weights':{'yolov8n_coco':'F:/Football-Fatigue-Project/models/yolov8n.pt'},
              'read_only_roots':['F:/Football-Fatigue-Project']}
    for entry in registry['matches'].values():entry.setdefault('calibration_dir',None)
    rp=ROOT/'configs/data_sources.yaml'
    if not rp.exists():rp.write_text(yaml.safe_dump(registry,sort_keys=False),encoding='utf-8')
    selected=deepcopy(m['candidate_configuration']);selected.pop('project',None)
    selected['detector']['operating_threshold']=.5
    selected['publication']={'size_model':'outputs/models/scratch_size_model.json',
                             'backgrounds':{'117093':'outputs/backgrounds/117093.png'}}
    (ROOT/'configs/scratch.yaml').write_text(yaml.safe_dump(selected,sort_keys=False),encoding='utf-8')
    for key,method,file,targetname in [('hog_svm','scratch_hog_svm','hog_svm.joblib','model.joblib'),('cnn','scratch_cnn','cnn.pt','model.pt')]:
        source=M1/'models'/file
        if sha(source)!=m['models'][key]['sha256']:raise ValueError('Original model hash mismatch')
        copy_new(source,ROOT/'outputs/models'/method/targetname)
    copy_new(M1/'size_model.json',ROOT/'outputs/models/scratch_size_model.json')
    ev=next(d for d in m['dataset'] if d['match']=='117093')
    copy_new(ROOT/ev['background'],ROOT/'outputs/backgrounds/117093.png')
    for name in ('train','val'):copy_new(M1/f'patches_{name}.npz',ROOT/f'data/processed/patches/patches_{name}.npz')
    current=cfg();sources=config.load_data_sources(); samples=[];stats={'schema_version':'1.0','created_utc':datetime.now(timezone.utc).isoformat(),'matches':{},'totals':dict(matches=0,frames=0,boxes=0)}
    for record in m['dataset']:
        mid=record['match'];paths=config.match_paths(mid,sources)
        if sha(paths.mot)!=record['annotation_sha256']:raise ValueError('Annotation hash changed')
        raw=data_io.read_mot_raw(paths.mot);gt,issues=data_io.mot_to_tracks(raw,(4096,1080),22)
        counts=gt.groupby('frame').size();lengths=gt.groupby('track_id').size()
        info=data_io.video_info(paths.clip)
        stats['matches'][mid]=dict(split=record['split'],image_width=info['width'],image_height=info['height'],fps=info['fps'],
            frames_annotated=int(raw.frame.nunique()),frames_header=info['frame_count_header'],
            frame_count_basis='header plus dense annotation agreement; no new full decode for EDA',boxes=len(gt),ids=int(gt.track_id.nunique()),
            boxes_per_frame=dict(min=int(counts.min()),max=int(counts.max()),mean=float(counts.mean())),
            box_width_px=data_io._box_quantiles(gt.w),box_height_px=data_io._box_quantiles(gt.h),
            track_length_frames=dict(min=int(lengths.min()),median=float(lengths.median()),max=int(lengths.max())),issues=issues)
        stats['totals']['matches']+=1;stats['totals']['frames']+=int(raw.frame.nunique());stats['totals']['boxes']+=len(gt)
        samples.append(gt.iloc[::17].assign(match=mid,split=record['split'])[['match','split','frame','x','y','w','h']])
    write(run/'eda/stats.json',stats);C.write_table(pd.concat(samples,ignore_index=True),run/'eda/box_sample.csv','box_sample')
    for name in ('stats.json','box_sample.csv'):copy_new(run/'eda'/name,ROOT/'outputs/eda'/name)
    from soccertrack.visualize import plot_eda
    plot_eda(stats,C.read_table(run/'eda/box_sample.csv','box_sample'),run/'figures/eda')
    dataset={}
    for name in ('train','val'):
        with np.load(M1/f'patches_{name}.npz',allow_pickle=False) as a:
            dataset[name]=dict(rows=len(a['y']),positive=int(a['y'].sum()),negative=int((a['y']==0).sum()),
                sampled_frames=len(set(zip(a['match'].tolist(),a['frame'].tolist()))),
                unique_pixels=len({hashlib.sha256(x.tobytes()).hexdigest() for x in a['X']}))
    for key,method in [('hog_svm','scratch_hog_svm'),('cnn','scratch_cnn')]:
        table=pd.read_csv(M1/f'validation_patch_scores_{key}.csv');p=ROOT/f'outputs/scores/{method}/val/patches.csv'
        meta=read(ROOT/'tests/fixtures/patch_scores_sample.meta.json')
        meta.update(method=method,config_sha256=config.config_sha256(),created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                    device='saved_predictions',model_sha256=m['models'][key]['sha256'],runtime_seconds=None,
                    notes='Saved expanded-run scores; repeated proposal negatives disclosed; no retraining')
        C.write_table(table,p,'patch_scores');C.write_run_meta(meta,p)
    write(run/'model_references.json',{'models':m['models'],'dataset':dataset,'source_run':str(M1.relative_to(ROOT)),
          'source_manifest_sha256':sha(M1/'manifest.json'),'config_sha256':config.config_sha256(),'HOG_reference':'unresolved API limitation'})
    from demos.member2.migrate_cache import migrate
    paths=migrate(M2,run/'yolo_migration',current,config.config_sha256(),read(M2/'manifest.json')['weights_sha256'])
    for name,p in paths.items():
        target=config.artifact_path('detections','yolov8n_coco','val','117093',name)
        copy_new(p,target);copy_new(C.meta_path(p),C.meta_path(target))
    mark(run,'publish')

def detection_meta(method,frames,total,seconds,notes):
    result = dict(schema_version='1.0',kind='detections',method=method,tracker=None,split='val',match='117093',
        frameset='eval10',frames=dict(start=frames.start,stop=frames.stop,stride=frames.step),source_num_frames=total,
        processed_frame_count=len(frames),image_width=4096,image_height=1080,fps=25.,score_floor=.05,operating_threshold=.5,
        frozen_test=False,config_sha256=config.config_sha256(),git_commit=None,
        created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),runtime_seconds=seconds,
        frames_per_second=len(frames)/seconds if seconds else None,device='cpu',runtime_scope='shared decode/proposals plus scorer; background/model setup/write excluded',notes=notes)
    if method in ('scratch_hog_svm','scratch_cnn'):
        result['model_sha256']=sha(ROOT/'outputs/models'/method/('model.joblib' if method=='scratch_hog_svm' else 'model.pt'))
    return result

def scratch(run):
    import torch
    torch.set_num_threads(2);current=cfg();sc=current['scratch'];sources=config.load_data_sources()
    paths=config.match_paths('117093',sources);total=data_io.video_info(paths.clip)['frame_count_header']
    ranges={name:config.frameset_range(name,total,current) for name in ('eval10','win60s')}
    ids=sorted(set(ranges['eval10'])|set(ranges['win60s']));bg=cv2.imread(str(ROOT/sc['publication']['backgrounds']['117093']))
    pitch=segmentation.pitch_mask(bg,sc,'117093',None)
    sm=proposals.SizeModel(**read(ROOT/sc['publication']['size_model']))
    models={'scratch_mwb0':None,'scratch_hog_svm':classifiers.load_classifier('scratch_hog_svm'),'scratch_cnn':classifiers.load_classifier('scratch_cnn')}
    starting=read(M1/'configuration.json');starting['detector']['operating_threshold']=.5
    rows={k:[] for k in models};props={k:[] for k in models};seconds={k:0. for k in models};starting_rows=[];seen=[];t0=time.perf_counter()
    checkpoint=run/'scratch_checkpoint.pkl';prior_elapsed=0.
    if checkpoint.exists():
        state=pd.read_pickle(checkpoint)
        if state['config_sha256']!=config.config_sha256():raise ValueError('Checkpoint configuration differs')
        rows,props,seconds,starting_rows,seen,prior_elapsed=(state[k] for k in ('rows','props','seconds','starting_rows','seen','elapsed'))
    journal(run,'Scratch union started:1948 frames; three existing methods; no training; shared decode/proposals')
    for index,frame in data_io.iter_frames(paths.clip,[i for i in ids if i not in set(seen)]):
        tp=time.perf_counter();base,raw,fg=D.prepare_frame_proposals(frame,index,bg,pitch,sm,sc);common=time.perf_counter()-tp
        for method,model in models.items():
            st=time.perf_counter();p,d=D.classify_candidates(frame,base.copy(),sc,model,.05)
            seconds[method]+=common+time.perf_counter()-st;rows[method].append(d)
            if index in ranges['eval10']:props[method].append(p)
        if index in ranges['eval10']:
            pp,_,_=D.prepare_frame_proposals(frame,index,bg,pitch,sm,starting)
            pp,_=D.classify_candidates(frame,pp,starting,models['scratch_hog_svm'],.05);starting_rows.append(pp)
        if index in (0,500,1000):
            folder=run/'stages';folder.mkdir(exist_ok=True)
            cv2.imwrite(str(folder/f'{index:06d}_frame.png'),frame)
            cv2.imwrite(str(folder/f'{index:06d}_foreground.png'),raw.astype(np.uint8)*255)
            cv2.imwrite(str(folder/f'{index:06d}_morph.png'),fg.astype(np.uint8)*255)
            from soccertrack.visualize import draw_boxes
            cv2.imwrite(str(folder/f'{index:06d}_proposals.png'),draw_boxes(frame,base,index))
            cv2.imwrite(str(folder/f'{index:06d}_pitch.png'),cv2.resize(pitch.astype(np.uint8)*255,(4096,1080),interpolation=cv2.INTER_NEAREST))
        seen.append(index)
        if len(seen)%100==0:
            print('scratch',len(seen),'/',len(ids),round(prior_elapsed+time.perf_counter()-t0,1),flush=True)
            pd.to_pickle(dict(rows=rows,props=props,seconds=seconds,starting_rows=starting_rows,seen=seen,
                elapsed=prior_elapsed+time.perf_counter()-t0,config_sha256=config.config_sha256()),checkpoint)
    if seen!=ids:raise RuntimeError('Incomplete union decoding')
    elapsed=prior_elapsed+time.perf_counter()-t0
    for method in models:
        union=pd.concat(rows[method],ignore_index=True)
        for name,frames in ranges.items():
            d=union[union.frame.isin(frames)].reset_index(drop=True)
            meta=detection_meta(method,frames,total,seconds[method]*len(frames)/len(ids),'Selected candidate; offline background; shared preparation attribution; union method time allocated proportional to frame count')
            meta['frameset']=name;meta['runtime_scope']='shared preparation + scorer processing estimate; proportional union allocation; decode excluded'
            target=config.artifact_path('detections',method,'val','117093',name)
            C.write_table(d,target,'detections');C.write_run_meta(meta,target)
        p=pd.concat(props[method],ignore_index=True);frames=ranges['eval10']
        target=config.artifact_path('proposals',method,'val','117093','eval10')
        meta=detection_meta(method,frames,total,None,'Every pre-NMS proposal; selected candidate; scores retained')
        meta.update(kind='proposals',runtime_seconds=None,frames_per_second=None)
        C.write_table(p,target,'proposals');C.write_run_meta(meta,target)
    startp=run/'variants/starting/proposals.csv';C.write_table(pd.concat(starting_rows,ignore_index=True),startp,'proposals')
    meta=detection_meta('scratch_hog_svm',ranges['eval10'],total,None,'Starting segmentation/proposals; same existing trained HOG classifier; supplementary controlled variant')
    meta.update(kind='proposals',runtime_seconds=None,frames_per_second=None);C.write_run_meta(meta,startp)
    write(run/'coverage.json',dict(frame_ids=seen,decoded_union_frames=len(seen),source_num_frames=total,
          union_wall_seconds=elapsed,processing_seconds_by_method=seconds,config_sha256=config.config_sha256()))
    mark(run,'scratch')

def evaluate(run):
    current=cfg();sources=config.load_data_sources();gt=data_io.load_gt_tracks('117093',sources);docs=[]
    for method in ('scratch_hog_svm','scratch_cnn'):
        digest=sha(ROOT/'outputs/models'/method/('model.joblib' if method=='scratch_hog_svm' else 'model.pt'))
        for kind,names in [('detections',['eval10','win60s']),('proposals',['eval10'])]:
            for name in names:
                path=config.artifact_path(kind,method,'val','117093',name)
                meta=C.read_run_meta(path)
                if meta['config_sha256']!=config.config_sha256():raise ValueError('Config changed during inference')
                C.write_run_meta(dict(meta,model_sha256=digest),path)
    for method in ('scratch_hog_svm','scratch_cnn'):
        docs.append(E.evaluate_patches(ROOT/f'outputs/scores/{method}/val/patches.csv',current))
        docs.append(E.evaluate_proposals(config.artifact_path('proposals',method,'val','117093','eval10'),gt,current))
    for method in ('scratch_mwb0','scratch_hog_svm','scratch_cnn','yolov8n_coco'):
        docs.append(E.evaluate_detections(config.artifact_path('detections',method,'val','117093','eval10'),gt,current))
    variant=E.evaluate_proposals(run/'variants/starting/proposals.csv',gt,current)
    variant['params']['experiment_variant']='starting';write(run/'variants/starting/metrics.json',variant)
    oracle=E.write_oracle_detections(gt,'val','117093','win60s',range(1500),(4096,1080),25.,current)
    best=max((d for d in docs if d['eval_id']=='E3' and d['method'].startswith('scratch')),key=lambda d:d['metrics']['f1'])['method']
    combinations=[('gt_oracle','iou_hungarian'),('gt_oracle','bytetrack'),('gt_noisy','iou_hungarian'),('gt_noisy','bytetrack'),(best,'iou_hungarian'),('yolov8n_coco','bytetrack')]
    for method,tracker in combinations:
        source=config.artifact_path('detections',method,'val','117093','win60s')
        det=C.read_table(source,'detections');meta=C.read_run_meta(source);settings=deepcopy(current['scratch'])
        settings['tracker_iou']['min_score']=meta['operating_threshold']
        target=config.artifact_path('tracks',C.pipeline_id(method,tracker),'val','117093','win60s')
        if method in (best,'yolov8n_coco'):
            import subprocess
            script='scripts/run_yolo.py' if method=='yolov8n_coco' else 'scripts/run_scratch.py'
            cmd=[sys.executable,'-B',script,'track','--detector',method,'--split','val','--match','117093','--frameset','win60s']
            process=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            (run/f'production_tracking_{method}.log').write_text(process.stdout+process.stderr,encoding='utf-8')
            if process.returncode:raise RuntimeError('Production tracking command failed')
        else:
            t=time.perf_counter()
            tracks=tracker_iou.track_sequence(det,range(1500),settings) if tracker=='iou_hungarian' else tracker_bytetrack.track_sequence(det,range(1500),current['yolo'])
            seconds=time.perf_counter()-t
            C.write_table(tracks,target,'tracks');C.write_run_meta(dict(meta,kind='tracks',tracker=tracker,runtime_seconds=seconds,
                frames_per_second=1500/max(seconds,1e-9),runtime_scope='association only; serialized detections; decode/inference/write excluded',
                source_detection_sha256=sha(source)),target)
        docs.append(E.evaluate_tracking(target,gt,current,workspace_root=run/'trackeval'))
        print('E4',method,tracker,docs[-1]['metrics'],flush=True)
    for doc in docs:E.write_metrics(doc,run/'metrics')
    compare.write_comparison(run/'metrics','val',run/'tables',run/'figures/eval',run/'eda/stats.json')
    artifacts={}
    for kind,methods,names in [('detections',['scratch_mwb0','scratch_hog_svm','scratch_cnn','yolov8n_coco'],['eval10','win60s']),
                               ('proposals',['scratch_mwb0','scratch_hog_svm','scratch_cnn'],['eval10']),
                               ('tracks',[C.pipeline_id(d,t) for d,t in combinations],['win60s'])]:
        for method in methods:
            for name in names:
                source=config.artifact_path(kind,method,'val','117093',name)
                target=run/'artifacts'/kind/method/f'117093__{name}.csv'
                copy_new(source,target);copy_new(C.meta_path(source),C.meta_path(target))
                artifacts[f'{kind}/{method}/{name}']=dict(path=target.relative_to(run).as_posix(),sha256=sha(target))
    from soccertrack import metrics as M
    labels=pd.read_csv(ROOT/'outputs/scores/scratch_hog_svm/val/patches.csv').y_true.to_numpy()
    dummy=M.binary_classification_metrics(labels,np.zeros(len(labels)),.5)
    write(run/'accepted_results.json',{'config_sha256':config.config_sha256(),'selected_scratch_detector':best,'results':docs,'artifacts':artifacts,'auxiliary_majority_baseline':dummy,
          'starting_E2':variant,'scope':'Integrated validation117093; no held-out results; HOG reference limitation disclosed'})
    mark(run,'evaluate')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('stage',choices=['publish','scratch','evaluate']);a=p.parse_args()
    run=a.run.resolve()
    if not run.is_relative_to(ROOT/'outputs/phase2_final'):raise ValueError('Versioned final outputs required')
    run.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(run/'mpl'));os.environ.setdefault('YOLO_CONFIG_DIR',str(run/'yolo'))
    state=read(run/'stage.json') if (run/'stage.json').exists() else {}
    if a.stage in state:print('Already completed:',a.stage);return
    globals()[a.stage](run)
if __name__=='__main__':main()
