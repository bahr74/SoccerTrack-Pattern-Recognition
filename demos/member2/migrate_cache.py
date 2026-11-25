"""Cache-only migration after shared integration; no inference, training or data reads.

Callers supply the real shared config and hash. This module does not implement
Member4's registry/config API and never labels historical raw tracks official.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import re
import numpy as np
import pandas as pd
from soccertrack import contracts as C, evaluate as E, split
from demos.member2.audit import digest
from demos.member2.safety import output_directory, ROOT

def verify_cache(run, cfg, expected_weights_sha256):
    run=Path(run).resolve()
    m=json.loads((run/'manifest.json').read_text())
    if m['split']!='val' or m['match']!='117093' or split.split_of(m['match'],cfg)!='val':
        raise PermissionError('Only approved validation cache migration is supported')
    if m['weights_sha256']!=expected_weights_sha256:
        raise ValueError('Model identity mismatch')
    if m['selected_configuration']!=cfg['yolo'] or m['dataset_revision']!=cfg['project']['dataset']['revision']:
        raise ValueError('Configuration/preprocessing/dataset revision mismatch; do not reuse cache')
    if not m.get('clip_sha256') or not m.get('mot_sha256'):
        raise ValueError('Missing source data provenance')
    required={'src/soccertrack/detector_yolo.py','src/soccertrack/tracker_bytetrack.py'}
    if not required.issubset(m['source_sha256']): raise ValueError('Missing detector/tracker source provenance')
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for rel in ['required_validation/detections.csv']:
        if rel not in hashes or digest(run/rel)!=hashes[rel]:
            raise ValueError('Prediction cache hash mismatch')
    for rel, expected in m['source_sha256'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Invalid archived source path')
        archived=run/'source_snapshot'/rel
        if digest(archived)!=expected: raise ValueError('Archived source hash mismatch')
        if rel in ('src/soccertrack/detector_yolo.py','src/soccertrack/tracker_bytetrack.py'):
            normalize=lambda p: p.read_text(encoding='utf-8-sig').replace('\r\n','\n')
            if normalize(archived)!=normalize(ROOT/rel): raise ValueError('Detector/tracker implementation changed')
    n=int(m['video_info']['frame_count_header'])
    frames={}
    for name in ('eval10','win60s'):
        spec=cfg['evaluation']['framesets'][name]
        stop=n if spec['stop'] is None else min(n,spec['stop'])
        ids=list(range(spec['start'],stop,spec['stride']))
        rel=f'required_validation/{name}/coverage.json'
        if rel not in hashes or digest(run/rel)!=hashes[rel]: raise ValueError('Coverage hash mismatch')
        coverage=json.loads((run/rel).read_text())
        if coverage['frame_ids']!=ids or coverage['decoded_frames']!=len(ids):
            raise ValueError('Incomplete cache coverage')
        frames[name]=range(spec['start'],stop,spec['stride'])
    union=sorted(set().union(*(set(r) for r in frames.values())))
    if m['required_union_frame_ids']!=union or m['decoded_union_frames']!=len(union):
        raise ValueError('Union frame coverage mismatch')
    if set(union)&set(m['development_frame_ids']): raise ValueError('Development/reporting leakage')
    table=pd.read_csv(run/'required_validation/detections.csv')
    C.validate_detections(table)
    if not set(table.frame).issubset(union): raise ValueError('Rows outside declared coverage')
    if len(table) and table.score.min()<cfg['evaluation']['score_floor']:
        raise ValueError('Score floor mismatch')
    return m,table,frames

def migrate(run, destination, cfg, shared_config_sha256, expected_weights_sha256):
    """Prepare official-format inputs; full E3/E4 execution remains a separate command."""
    if not re.fullmatch('[0-9a-f]{64}',shared_config_sha256) or shared_config_sha256=='0'*64:
        raise ValueError('Actual Member4 shared config hash required; no placeholder promotion')
    destination=output_directory(destination,fresh=True)
    m,table,frames=verify_cache(run,cfg,expected_weights_sha256)
    info=m['video_info']; written={}
    # All checks above precede any output creation. Fresh destinations prevent overwrites.
    for name,r in frames.items():
        p=destination/'detections/yolov8n_coco/val'/f'117093__{name}.csv'
        selected=table[table.frame.isin(r)].reset_index(drop=True)
        meta=dict(schema_version=C.SCHEMA_VERSION,kind='detections',method='yolov8n_coco',tracker=None,
                  split='val',match='117093',frameset=name,frames=dict(start=r.start,stop=r.stop,stride=r.step),
                  source_num_frames=int(info['frame_count_header']),processed_frame_count=len(r),
                  image_width=int(info['width']),image_height=int(info['height']),fps=float(info['fps']),
                  score_floor=cfg['evaluation']['score_floor'],operating_threshold=cfg['yolo']['operating_threshold'],
                  frozen_test=False,config_sha256=shared_config_sha256,git_commit=m['git_commit'],
                  created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                  runtime_seconds=None,frames_per_second=None,device='cached',runtime_scope='cache migration; no inference timing',
                  weights_sha256=expected_weights_sha256,dataset_revision=m['dataset_revision'],
                  source_table_sha256=digest(Path(run)/'required_validation/detections.csv'),
                  notes='Cache migration; E3/E4 pending; pitch_filter=off; original runtime stays in source manifest')
        E.validate_protocol(meta,cfg,'detections')
        C.validate_detections(selected,meta)
        C.write_table(selected,p,'detections'); C.write_run_meta(meta,p)
        C.validate_detections(C.read_table(p,'detections'),C.read_run_meta(p))
        written[name]=p
    (destination/'migration_manifest.json').write_text(json.dumps({
        'status':'INPUTS MIGRATED; official evaluation and tracker replay pending',
        'source_manifest_sha256':digest(Path(run)/'manifest.json'),'shared_config_sha256':shared_config_sha256,
        'tables':{k:{'path':v.relative_to(destination).as_posix(),'sha256':digest(v)} for k,v in written.items()},
        'inference_rerun':False,'raw_tracks_reused':False},indent=2)+'\n',encoding='utf-8')
    return written

def replay(detections_csv, target, cfg):
    """Replay association from read-back serialized detections, never from raw historical tracks."""
    from soccertrack.tracker_bytetrack import track_sequence
    import time
    source=Path(detections_csv);target=Path(target)
    output_directory(target.parent)
    if target.exists() or C.meta_path(target).exists(): raise FileExistsError(target)
    meta=C.read_run_meta(source);E.validate_protocol(meta,cfg,'detections')
    if meta['frames']['stride']!=1: raise ValueError('Tracker replay requires continuous stride1')
    table=C.read_table(source,'detections');C.validate_detections(table,meta)
    frames=range(meta['frames']['start'],meta['frames']['stop'],1)
    start=time.perf_counter(); tracks=track_sequence(table,frames,cfg['yolo']);elapsed=time.perf_counter()-start
    updated=dict(meta,kind='tracks',tracker='bytetrack',runtime_seconds=elapsed,
                 frames_per_second=len(frames)/elapsed if elapsed else None,
                 created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                 runtime_scope='ByteTrack association only; source read/decode/inference/write excluded',
                 source_detection_sha256=digest(source),notes='Replay from canonical serialized detections; E4 pending')
    C.validate_tracks(tracks,(meta['image_width'],meta['image_height']))
    C.write_table(tracks,target,'tracks');C.write_run_meta(updated,target)
    C.read_table(target,'tracks')
    return target
