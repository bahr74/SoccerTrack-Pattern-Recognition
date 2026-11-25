"""Member 2 CLI. Official shared interfaces are required for artifact publication."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import time
import re
import subprocess
from soccertrack import config,contracts
from soccertrack.split import require_test_access,matches_for,split_of,log_test_access
from soccertrack.detector_yolo import run_detection
from soccertrack.tracker_bytetrack import track_sequence
COMMANDS=('detect','track','finetune')

def build_parser():
    parser=argparse.ArgumentParser(description='Approach B tiled YOLOv8n and ByteTrack')
    parser.add_argument('command',choices=COMMANDS)
    parser.add_argument('--split',choices=('train','val','test'),default='val')
    parser.add_argument('--match',default=None)
    parser.add_argument('--frameset',choices=('eval10','win60s','full','smoke'),default='eval10')
    parser.add_argument('--detector',default='yolov8n_coco')
    parser.add_argument('--max-frames',type=int,default=None)
    parser.add_argument('--frozen-test',action='store_true')
    return parser

def main():
    args=build_parser().parse_args()
    require_test_access(args.split,args.frozen_test)
    if args.command=='finetune':
        raise SystemExit('Phase 3 fine-tuning is outside current authorization')
    if args.max_frames is not None and args.max_frames<=0:
        raise SystemExit('--max-frames must be positive')
    cfg=config.load_config()
    sources=config.load_data_sources()
    config.set_seeds(cfg['project']['seed'])
    matches=[args.match] if args.match else matches_for(args.split,cfg)
    if args.max_frames is not None:
        cfg=deepcopy(cfg)
        cfg['_member2_max_frames']=args.max_frames
    for match in matches:
        if split_of(match,cfg)!=args.split:
            raise ValueError('match/split mismatch')
        if args.command=='detect':
            run_detection(args.detector,match,args.split,args.frameset,cfg,sources,args.frozen_test)
            continue
        input_path=config.artifact_path('detections',args.detector,args.split,match,args.frameset)
        meta=contracts.read_run_meta(input_path)
        if any(meta[k]!=v for k,v in [('method',args.detector),('match',match),('split',args.split),('frameset',args.frameset)]):
            raise ValueError('input metadata differs from requested sequence')
        if args.split=='test' and not meta['frozen_test']:
            raise ValueError('test input must be frozen')
        spec=meta['frames']
        frames=range(spec['start'],spec['stop'],spec['stride'])
        if frames.step!=1:
            raise ValueError('ByteTrack requires stride 1; eval10 cannot be tracked')
        detections=contracts.read_table(input_path,'detections')
        if args.max_frames is not None:
            frames=frames[:args.max_frames]
            detections=detections[detections.frame.isin(frames)]
        start=time.perf_counter()
        tracks=track_sequence(detections,frames,cfg)
        elapsed=time.perf_counter()-start
        # Kalman boxes must be clipped before writing a contract artifact.
        for axis,size in [('x',meta['image_width']),('y',meta['image_height'])]:
            extent='w' if axis=='x' else 'h'
            end=(tracks[axis]+tracks[extent]).clip(0,size)
            tracks[axis]=tracks[axis].clip(0,size)
            tracks[extent]=end-tracks[axis]
        tracks=tracks[(tracks.w>1)&(tracks.h>1)].reset_index(drop=True)
        pipeline=contracts.pipeline_id(args.detector,'bytetrack')
        output=config.artifact_path('tracks',pipeline,args.split,match,args.frameset)
        if args.max_frames is not None:
            output=config.repo_root()/'outputs/debug/tracks'/pipeline/args.split/f'{match}__{args.frameset}.csv'
            output.parent.mkdir(parents=True,exist_ok=True)
        updated=dict(meta,kind='tracks',tracker='bytetrack',frames={'start':frames.start,'stop':frames.stop,'stride':1},
            input_detection_runtime_seconds=meta.get('runtime_seconds'),
            input_detection_runtime_scope=meta.get('runtime_scope','source detection notes'),
            runtime_scope='ByteTrack association only; model/decode/write excluded',
            runtime_seconds=elapsed,frames_per_second=len(frames)/elapsed if elapsed else 0,device='cpu',
            config_sha256=config.config_sha256(),created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=config.repo_root(),text=True).strip(),
            notes=f'ByteTrack association only; input low-score detections retained; model/decode/write excluded; input_config_sha256={meta["config_sha256"]}; input_table={input_path.relative_to(config.repo_root()).as_posix()}; source_detection_notes='+meta['notes'])
        contracts.validate_tracks(tracks,(meta['image_width'],meta['image_height']))
        contracts.validate_run_meta(updated)
        contracts.write_table(tracks,output,'tracks')
        contracts.write_run_meta(updated,output)
        if args.split=='test':
            match_hash=re.search(r'weights_sha256=([0-9a-f]{64})',meta['notes'])
            hashes={args.detector:match_hash.group(1)} if match_hash else {}
            log_test_access('Member 2',pipeline,[output.relative_to(config.repo_root()).as_posix()],hashes)

if __name__=='__main__':
    main()
