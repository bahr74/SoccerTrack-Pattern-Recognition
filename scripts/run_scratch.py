"""Member 1 entry point: Approach A (from scratch) and Pipeline A tracking.

Owner: Member 1. Usage (PowerShell, from the repository root, with .venv active):
  python scripts/run_scratch.py eda
  python scripts/run_scratch.py segment     --split val
  python scripts/run_scratch.py detect      --detector scratch_mwb0 --split val --frameset eval10
  python scripts/run_scratch.py patches     --split train
  python scripts/run_scratch.py train-svm
  python scripts/run_scratch.py train-cnn
  python scripts/run_scratch.py score-patches --detector scratch_hog_svm --split val
  python scripts/run_scratch.py detect      --detector scratch_hog_svm --split val --frameset eval10
  python scripts/run_scratch.py track       --detector gt_oracle --split val --frameset win60s
  python scripts/run_scratch.py track       --detector scratch_hog_svm --split val --frameset win60s
Test split: add --frozen-test (only during the logged frozen test run).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
from datetime import datetime, timezone

from soccertrack import config, contracts, data_io, split, patches, classifiers, detector_scratch, tracker_iou, segmentation

COMMANDS = ("eda", "segment", "patches", "train-svm", "train-cnn", "score-patches", "detect", "track")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Approach A: from-scratch detection and IoU+Hungarian tracking")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--match", default=None, help="one match ID; default = every match of --split")
    parser.add_argument("--frameset", choices=("eval10", "win60s", "full", "smoke"), default="eval10")
    parser.add_argument("--detector", default="scratch_hog_svm",
                        help="scratch_mwb0 | scratch_hog_svm | scratch_cnn (track also accepts gt_oracle, gt_noisy)")
    parser.add_argument("--frozen-test", action="store_true", help="required for --split test")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split.require_test_access(args.split, args.frozen_test)
    cfg = config.load_config()  # official runner never falls back to demo YAML loading
    root = config.repo_root()
    outputs = root / cfg['project']['paths']['outputs_root']
    if args.command == 'eda':
        if args.split == 'test': raise PermissionError('EDA does not access held-out data')
        sources = config.load_data_sources()
        splits = {name: split.matches_for(name, cfg) for name in ('train','val')}
        stats = data_io.compute_eda_stats(splits, sources)
        folder = outputs / 'eda'; folder.mkdir(parents=True, exist_ok=True)
        (folder / 'stats.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if 'box_sample' not in contracts.TABLE_SCHEMAS:
            raise NotImplementedError('stats.json computed; box_sample.csv requires a registered Member 3 table kind')
        import pandas as pd
        samples=[]
        for match, item in stats['matches'].items():
            frame=data_io.load_gt_tracks(match,sources).iloc[::7].assign(match=match,split=item['split'])
            samples.append(frame[['match','split','frame','x','y','w','h']])
        if samples: contracts.write_table(pd.concat(samples,ignore_index=True),folder/'box_sample.csv','box_sample')
        return
    if args.command in ('train-svm','train-cnn'):
        train = _load_patches('train', cfg)
        val = _load_patches('val', cfg)
        sc = dict(cfg['scratch'], project=cfg['project'])
        if args.command == 'train-svm':
            model=classifiers.HogSvmClassifier(sc).fit_with_validation(train['X'],train['y'],val['X'],val['y'])
            model.save(outputs/'models/scratch_hog_svm/model.joblib')
        else:
            model=classifiers.CnnClassifier(sc)
            history=model.fit(train['X'],train['y'],val['X'],val['y'])
            folder=outputs/'models/scratch_cnn'; model.save(folder/'model.pt')
            (folder/'history.json').write_text(json.dumps(history,indent=2)+'\n',encoding='utf-8')
        return
    if args.command == 'score-patches':
        if args.match is not None:
            raise ValueError('score-patches exports the entire split, not a single match')
        arrays=_load_patches(args.split,cfg)
        model=classifiers.load_classifier(args.detector)
        import hashlib
        checkpoint=outputs/'models'/args.detector/('model.joblib' if args.detector=='scratch_hog_svm' else 'model.pt')
        target=outputs/'scores'/args.detector/args.split/'patches.csv'
        export_patch_scores(arrays,model,args.detector,args.split,cfg,target,
                            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),config.config_sha256(),args.frozen_test)
        if args.split=='test': split.log_test_access('Member 1',args.detector,[target.relative_to(root).as_posix()],{args.detector:hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        return
    sources=config.load_data_sources()
    if args.command == 'patches':
        arrays=patches.build_patch_set(args.split,cfg,sources,args.frozen_test)
        path=root/cfg['project']['paths']['data_root']/f'processed/patches/patches_{args.split}.npz'
        patches.save_patch_set(arrays,path)
        if args.split=='test': split.log_test_access('Member 1','patches',[path.relative_to(root).as_posix()],{})
        return
    matches=[args.match] if args.match else split.matches_for(args.split,cfg)
    for match in matches:
        if split.split_of(match,cfg)!=args.split: raise ValueError('Match/split mismatch')
        if args.command=='detect':
            detector_scratch.run_detection(args.detector,match,args.split,args.frameset,cfg,sources,args.frozen_test)
        elif args.command=='track':
            source=config.artifact_path('detections',args.detector,args.split,match,args.frameset)
            detections=contracts.read_table(source,'detections'); meta=contracts.read_run_meta(source)
            frames=range(meta['frames']['start'],meta['frames']['stop'],meta['frames']['stride'])
            settings=dict(cfg['scratch']); settings['tracker_iou']=dict(settings['tracker_iou'])
            if settings['tracker_iou']['min_score'] is None: settings['tracker_iou']['min_score']=meta['operating_threshold']
            start=time.perf_counter(); tracks=tracker_iou.track_sequence(detections,frames,settings); seconds=time.perf_counter()-start
            pipeline=contracts.pipeline_id(args.detector,'iou_hungarian')
            target=config.artifact_path('tracks',pipeline,args.split,match,args.frameset)
            contracts.write_table(tracks,target,'tracks')
            contracts.write_run_meta(dict(meta,kind='tracks',tracker='iou_hungarian',runtime_seconds=seconds,
                                           config_sha256=config.config_sha256(),
                                           created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
                                           frames_per_second=len(frames)/max(seconds,1e-9)),target)
            if args.split=='test': split.log_test_access('Member 1',pipeline,[target.relative_to(root).as_posix()],{})
        elif args.command=='segment':
            import cv2
            paths=config.match_paths(match,sources); info=data_io.video_info(paths.clip)
            count=data_io.count_decoded_frames(paths.clip); sc=cfg['scratch']; scale=sc['segmentation']['work_scale']
            background=segmentation.estimate_background(paths.clip,sc['sampling']['background_num_frames'],count,scale)
            mask=segmentation.pitch_mask(background,sc,match,paths.calibration_dir)
            segmentation.save_pitch_mask(mask,match,(info['width'],info['height']))
            folder=outputs/'backgrounds'; folder.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(folder/f'{match}.png'),background): raise OSError('Background write failed')
        else: raise ValueError(f'Unsupported command {args.command}')


def _load_patches(name, cfg):
    root=config.repo_root()
    arrays=patches.load_patch_set(root/cfg['project']['paths']['data_root']/f'processed/patches/patches_{name}.npz')
    meta=json.loads(arrays['meta_json'][0])
    if meta['split']!=name or not set(arrays['match']).issubset(split.matches_for(name,cfg)):
        raise PermissionError('Patch-set split/provenance mismatch')
    return arrays


def export_patch_scores(arrays, model, detector_id, split_name, cfg, target, model_sha256, config_sha256, frozen_test=False):
    """Aggregate score export with the existing nullable sidecar; no fitting or path infrastructure."""
    import pandas as pd
    import re
    split.require_test_access(split_name, frozen_test)
    contracts.validate_patch_set(arrays)
    if detector_id not in ('scratch_hog_svm','scratch_cnn'):
        raise ValueError('Patch scoring requires a registered scratch classifier')
    if not set(arrays['match']).issubset(split.matches_for(split_name,cfg)):
        raise PermissionError('Patch-set split/provenance mismatch')
    if not re.fullmatch('[0-9a-f]{64}',model_sha256): raise ValueError('Model SHA256 required')
    target=Path(target)
    if target.exists() or contracts.meta_path(target).exists(): raise FileExistsError(target)
    start=time.perf_counter()
    scores=model.predict_proba(arrays['X'])
    table=pd.DataFrame(dict(sample_id=range(len(scores)),match=arrays['match'],frame=arrays['frame'],
                            kind=arrays['kind'],y_true=arrays['y'],score=scores))
    threshold=cfg['scratch']['detector'].get('operating_threshold')
    if threshold is None: threshold=.5  # approved E1 classifier threshold, not a detector selection
    meta=dict(schema_version=contracts.SCHEMA_VERSION,kind='patch_scores',method=detector_id,tracker=None,
              split=split_name,match=None,frameset=None,frames=None,image_width=None,image_height=None,fps=None,
              score_floor=None,operating_threshold=threshold,frozen_test=frozen_test,config_sha256=config_sha256,
              git_commit=None,created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
              runtime_seconds=time.perf_counter()-start,frames_per_second=None,device='cached_model',
              model_sha256=model_sha256,runtime_scope='patch scoring only; dataset load/write excluded',
              notes='Aggregate patch scores; model_sha256='+model_sha256+'; nominal rows may include repeated proposal negatives; preserve source sampling disclosure')
    contracts.validate_patch_scores(table); contracts.validate_run_meta(meta)
    contracts.write_table(table,target,'patch_scores'); contracts.write_run_meta(meta,target)
    contracts.read_table(target,'patch_scores'); contracts.read_run_meta(target)
    return target


if __name__ == "__main__":
    main()
