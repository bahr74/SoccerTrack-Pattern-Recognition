"""Finish the already-planned Member2 three-candidate validation comparison, cache-only."""
from pathlib import Path
from copy import deepcopy
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from demos.phase2.workflow import read,write,copy_new,journal,sha
from soccertrack import config,contracts as C,data_io,evaluate as E,compare,tracker_bytetrack
def select(run):
    run=Path(run).resolve();data=read(run/'accepted_results.json');cfg=config.load_config()
    journal(run,'Tracker selection predeclared: existing starting/conservative/long_buffer configurations only; maximum HOTA, then IDF1, then MOTA; no new detector inference or threshold search')
    source=config.artifact_path('detections','yolov8n_coco','val','117093','win60s');det=C.read_table(source,'detections');meta=C.read_run_meta(source);gt=data_io.load_gt_tracks('117093',config.load_data_sources())
    variants=run/'variants/tracker';variants.mkdir(parents=True,exist_ok=True)
    original=config.artifact_path('tracks','yolov8n_coco+bytetrack','val','117093','win60s')
    start=variants/'starting/tracks.csv';copy_new(original,start);copy_new(C.meta_path(original),C.meta_path(start))
    doc=next(d for d in data['results'] if d['eval_id']=='E4' and d['method']=='yolov8n_coco+bytetrack')
    candidates=[('starting',doc,start,deepcopy(cfg['yolo']['bytetrack']))]
    for name,changes in [('conservative',dict(track_high_thresh=.4,new_track_thresh=.4)),('long_buffer',dict(track_buffer=60,match_thresh=.9))]:
        local=deepcopy(cfg['yolo']);local['bytetrack'].update(changes)
        t=time.perf_counter();tracks=tracker_bytetrack.track_sequence(det,range(1500),local);elapsed=time.perf_counter()-t
        path=variants/name/'tracks.csv';C.write_table(tracks,path,'tracks')
        C.write_run_meta(dict(meta,kind='tracks',tracker='bytetrack',tracker_settings=local['bytetrack'],
            runtime_seconds=elapsed,frames_per_second=1500/elapsed,runtime_scope='association only from serialized detections',
            source_detection_sha256=sha(source),notes=f'Preplanned tracker candidate:{name}; base shared config plus explicit tracker override'),path)
        result=E.evaluate_tracking(path,gt,cfg,workspace_root=variants/name/'trackeval')
        write(variants/name/'metrics.json',result);write(variants/name/'tracker_settings.json',local['bytetrack'])
        print(name,result['metrics'],flush=True);candidates.append((name,result,path,local['bytetrack']))
    best=max(candidates,key=lambda item:tuple(item[1]['metrics'][k] for k in ('hota','idf1','mota')))
    name,selected,path,settings=best
    data['tracker_selection']={'criterion':'max HOTA, then IDF1, then MOTA; preplanned three candidates only','selected':name,
        'settings':settings,'candidates':{n:d['metrics'] for n,d,_,_ in candidates},'scope':'supplementary validation selection; no independent held-out assessment'}
    write(run/'selected_tracker_settings.json',settings)
    data['results']=[selected if d['eval_id']=='E4' and d['method']=='yolov8n_coco+bytetrack' else d for d in data['results']]
    # Preserve starting artifacts and use the selected immutable variant directly.
    data['artifacts']['tracks/yolov8n_coco+bytetrack/win60s']={'path':path.relative_to(run).as_posix(),'sha256':sha(path)}
    if path!=start:
        import shutil
        if sha(original)!=sha(start):raise ValueError('Canonical tracks changed outside this selection')
        shutil.copy2(path,original);shutil.copy2(C.meta_path(path),C.meta_path(original))
    E.write_metrics(selected,run/'metrics');write(run/'accepted_results.json',data)
    compare.write_comparison(run/'metrics','val',run/'tables',run/'figures/eval',run/'eda/stats.json')
    journal(run,'Preplanned tracker comparison completed; selected:'+name)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();select(a.run)
