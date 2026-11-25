"""Isolated, explicitly uncertified real validation workflow. No official artifacts."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import time
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.member2.audit import digest
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'outputs/member2_scientific/ultralytics')
os.environ['MPLCONFIGDIR']=str(ROOT/'outputs/member2_scientific/matplotlib')
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from soccertrack.data_io import iter_frames,video_info,read_mot_raw,clean_mot_geometry
from soccertrack.detector_yolo import YoloTiledDetector
from soccertrack.tracker_bytetrack import track_sequence

WEIGHTS=Path('F:/Football-Fatigue-Project/models/yolov8n.pt')
CLIP=Path('F:/Football-Fatigue-Project/data/external/soccertrack-v2/raw/117093/117093.mp4')
MOT=Path('F:/Football-Fatigue-Project/data/external/soccertrack-v2/mot/117093.txt')
EXPECTED='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
STATUS='PROVISIONAL LOCAL DIAGNOSTICS - shared validation/E3/E4 unavailable'

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def metrics(dets,gt,frames,threshold):
    """Local score-ordered greedy IoU .5 counts; not certified E3 or AP."""
    groups={int(f):g[['x','y','w','h']].to_numpy(float) for f,g in gt.groupby('frame') if f in frames}
    predictions={int(f):g.sort_values('score',ascending=False,kind='stable') for f,g in dets[dets.score>=threshold].groupby('frame')}
    tp=fp=0
    labels=[]
    for f in frames:
        truth=groups.get(f,np.empty((0,4)))
        available=np.ones(len(truth),bool)
        group=predictions.get(f)
        if group is None:
            labels.extend((f,'fn',box.tolist()) for box in truth)
            continue
        for row in group.itertuples(index=False):
            box=np.array([row.x,row.y,row.w,row.h])
            lo=np.maximum(truth[:,:2],box[:2])
            hi=np.minimum(truth[:,:2]+truth[:,2:],box[:2]+box[2:])
            area=np.maximum(hi-lo,0).prod(axis=1)
            union=truth[:,2:].prod(axis=1)+box[2:].prod()-area
            iou=np.divide(area,union,out=np.zeros_like(area),where=union>0)
            iou[~available]=-1
            index=int(iou.argmax()) if len(iou) else -1
            matched=index>=0 and iou[index]>=.5
            if matched:
                tp+=1
                available[index]=False
            else:
                fp+=1
            labels.append((f,'tp' if matched else 'fp',box.tolist()))
        labels.extend((f,'fn',box.tolist()) for box in truth[available])
    total=sum(len(groups.get(f,[])) for f in frames)
    fn=total-tp
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/total if total else 0
    return dict(status=STATUS,threshold=float(threshold),frames=len(frames),gt=total,pred=tp+fp,tp=tp,fp=fp,fn=fn,
                precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0),labels

def draw(image,boxes,color,text=None):
    out=image.copy()
    for i,box in enumerate(boxes):
        x,y,w,h=map(float,box[:4])
        cv2.rectangle(out,(round(x),round(y)),(round(x+w),round(y+h)),color,2)
        if text is not None:
            cv2.putText(out,str(text[i]),(round(x),max(15,round(y)-3)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    return out

def infer(detector,frames,out,save_stages=False):
    tables,timings,seen=[],[],[]
    begin=time.perf_counter()
    generator=iter_frames(CLIP,frames)
    try:
        for f,image in generator:
            table=detector.detect_frame(image,f)
            tables.append(table)
            timings.append({'frame':f,**detector.last_timing})
            seen.append(f)
            if save_stages and f==frames[0]:
                stages=out/'stages'
                stages.mkdir(exist_ok=True)
                cv2.imwrite(str(stages/'original.jpg'),image)
                debug=detector.last_debug
                tiled=image.copy()
                for x0,y0,x1,y1 in debug['tiles']:
                    cv2.rectangle(tiled,(x0,y0),(x1-1,y1-1),(255,0,255),2)
                cv2.imwrite(str(stages/'tiles.jpg'),tiled)
                for name in ('pre_nms','post_nms'):
                    boxes=debug[name].copy()
                    boxes[:,2:]-=boxes[:,:2]
                    cv2.imwrite(str(stages/f'{name}.jpg'),draw(image,boxes,(0,255,255)))
                x0,y0,x1,y1=debug['tiles'][0]
                box=debug['local_boxes'][0].copy()
                box[:,2:]-=box[:,:2]
                cv2.imwrite(str(stages/'tile_local.jpg'),draw(image[y0:y1,x0:x1],box,(0,255,255)))
                write_json(stages/'geometry.json',{'tiles':debug['tiles'],'local_boxes':[b.tolist() for b in debug['local_boxes']],
                    'pre_nms':debug['pre_nms'].tolist(),'post_nms':debug['post_nms'].tolist(),'frame':f,'status':STATUS})
            if len(seen)%100==0:
                print(f'{out.name}: {len(seen)}/{len(frames)} frames',flush=True)
    finally:
        generator.close()
    elapsed=time.perf_counter()-begin
    if seen!=list(frames):
        raise RuntimeError('incomplete decoded frame coverage')
    dets=pd.concat(tables,ignore_index=True)
    timing=pd.DataFrame(timings)
    dets.to_csv(out/'detections.csv',index=False,lineterminator='\n')
    timing.to_csv(out/'timings.csv',index=False,lineterminator='\n')
    write_json(out/'coverage.json',{'frame_ids':seen,'decoded_frames':len(seen),'status':STATUS})
    return dets,timing,elapsed

def run(output=None):
    if not WEIGHTS.is_file() or digest(WEIGHTS)!=EXPECTED or not CLIP.is_file() or not MOT.is_file():
        raise RuntimeError('approved local validation data/weights unavailable or hash mismatch')
    out=Path(output) if output else ROOT/'outputs/member2_scientific/runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    from demos.member2.environment import capture
    capture(WEIGHTS,EXPECTED)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    baseline=yaml.safe_load((ROOT/'configs/yolo.yaml').read_text(encoding='utf-8-sig'))
    evalcfg=yaml.safe_load((ROOT/'configs/evaluation.yaml').read_text())
    gt,issues=clean_mot_geometry(read_mot_raw(MOT),(4096,1080),22)
    # Kept as raw geometry: no shared contract certification is claimed.
    gt=gt.rename(columns={'id':'track_id'})
    development=list(range(1601,5762,130))
    configs={}
    for name,width,imgsz,overlap in [('reference',1280,960,64),('native1280',1280,1280,64),
                                  ('tile960',960,1280,64),('overlap128',1280,1280,128)]:
        cfg=deepcopy(baseline)
        cfg['tiling'].update(tile_width=width,imgsz=imgsz,overlap_px=overlap)
        configs[name]=cfg
    manifest={'status':STATUS,'run_id':out.name,'created_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:digest(ROOT/p) for p in ['src/soccertrack/detector_yolo.py','src/soccertrack/tracker_bytetrack.py','demos/member2/scientific_workflow.py','demos/member2/environment.py']},
        'weights_sha256':EXPECTED,'dataset_revision':'325bd4c57ee8e135ab652f3921d274526ada496b',
        'clip_sha256':digest(CLIP),'mot_sha256':digest(MOT),'split':'val','match':'117093','seed':42,
        'versions':{p:md.version(p) for p in ['ultralytics','torch','torchvision','numpy','pandas','opencv-python','lap']},
        'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','development_frame_ids':development,
        'starting_configuration':baseline,'candidates':configs,'evaluation_config':evalcfg,'pitch_filter':'off; no approved mask available',
        'gt_cleaning_issues':issues,'official_e3':'BLOCKED: Member 3 evaluator/contracts M0 placeholders',
        'official_e4':'BLOCKED: Member 3 evaluator/contracts M0 placeholders','oracle_noisy':'BLOCKED: Member 3 official producers unavailable',
        'timing_definition':'processing=tile copy+YOLO preprocess/inference/local NMS+CPU global merge, CUDA synchronized. predict includes first-call warmup. model load separate. end_to_end includes sequential decode and per-frame assembly, stages when enabled; excludes CSV writing/plots. No real-time claim.'}
    write_json(out/'manifest.json',manifest)
    rows=[]
    cache={}
    for name,cfg in configs.items():
        dest=out/name
        dest.mkdir()
        write_json(dest/'config.json',cfg)
        start=time.perf_counter()
        detector=YoloTiledDetector(cfg,WEIGHTS)
        loaded=time.perf_counter()-start
        dets,timing,elapsed=infer(detector,development,dest,save_stages=True)
        cache[name]=dets
        for threshold in np.round(np.arange(.05,.91,.05),2):
            result,_=metrics(dets,gt,development,threshold)
            result.update(configuration=name,processing_seconds=float(timing.processing_seconds.sum()),
                fps=len(development)/timing.processing_seconds.sum(),decode_included_seconds=elapsed,
                model_load_seconds=loaded,tiles_per_frame=len(detector.last_debug['tiles']),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
            rows.append(result)
        print(name, 'development complete',flush=True)
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    grid=pd.DataFrame(rows)
    grid.to_csv(out/'development_results.csv',index=False)
    # Quality first, measured runtime tie-break; same 33 development frames for all.
    best=grid.sort_values(['f1','processing_seconds'],ascending=[False,True],kind='stable').iloc[0]
    selected=deepcopy(configs[best.configuration])
    selected['operating_threshold']=float(best.threshold)
    manifest.update(selection_criterion='max local development F1 at IoU .5; processing time tie-break',
                    selected_configuration=selected,selected_candidate=str(best.configuration),selection_frames_overlap_report_frames=False)
    write_json(out/'selected_config.json',selected)
    print('Selected',best.configuration,best.threshold,best.f1,flush=True)
    dest=out/'required_validation'
    dest.mkdir()
    info=video_info(CLIP)
    eval10=list(range(0,info['frame_count_header'],10))
    win60s=list(range(1500))
    union=sorted(set(eval10+win60s))
    detector=YoloTiledDetector(selected,WEIGHTS)
    dets,timing,elapsed=infer(detector,union,dest,save_stages=True)
    summary=[]
    for name,frames in [('eval10',eval10),('win60s',win60s)]:
        part=dest/name
        part.mkdir()
        subset=dets[dets.frame.isin(frames)]
        subset.to_csv(part/'detections.csv',index=False)
        subset_timing=timing[timing.frame.isin(frames)]
        write_json(part/'coverage.json',{'frame_ids':frames,'decoded_frames':len(frames),'status':STATUS})
        result,_=metrics(subset,gt,frames,selected['operating_threshold'])
        result.update(frameset=name,processing_seconds=float(subset_timing.processing_seconds.sum()),
                      fps=len(frames)/subset_timing.processing_seconds.sum())
        summary.append(result)
    write_json(dest/'summary.json',summary)
    threshold_rows=[]
    for threshold in np.round(np.arange(.05,.91,.05),2):
        result,_=metrics(dets,gt,eval10,threshold)
        threshold_rows.append(result)
    pd.DataFrame(threshold_rows).to_csv(dest/'eval10_threshold_analysis.csv',index=False)
    trackrows=[]
    trackconfigs={'starting':deepcopy(selected),'conservative':deepcopy(selected),'long_buffer':deepcopy(selected)}
    trackconfigs['conservative']['bytetrack'].update(track_high_thresh=.4,new_track_thresh=.4)
    trackconfigs['long_buffer']['bytetrack'].update(track_buffer=60,match_thresh=.9)
    for name,cfg in trackconfigs.items():
        before=time.perf_counter()
        tracks=track_sequence(dets[dets.frame<1500].reset_index(drop=True),range(1500),cfg)
        elapsed_track=time.perf_counter()-before
        tracks.to_csv(dest/f'tracks_{name}.csv',index=False)
        lengths=tracks.groupby('track_id').size()
        trackrows.append({'status':STATUS,'configuration':name,'frames':1500,'ids':len(lengths),'observations':len(tracks),
            'median_observations_per_id':float(lengths.median()) if len(lengths) else 0,'max_observations_per_id':int(lengths.max()) if len(lengths) else 0,
            'association_seconds':elapsed_track,'association_fps':1500/elapsed_track,'settings':cfg['bytetrack']})
    write_json(dest/'tracking_diagnostics.json',trackrows)
    manifest.update(required_detection_results=summary,tracking_diagnostics=trackrows,
                    required_union_frame_ids=union,decoded_union_frames=len(union),union_decode_included_seconds=elapsed,
                    union_decode_included_fps=len(union)/elapsed,video_info=info,
                    config_sha256=digest(out/'selected_config.json'))
    write_json(out/'manifest.json',manifest)
    print('REAL VALIDATION COMPLETE',out,flush=True)
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    run(args.output)
