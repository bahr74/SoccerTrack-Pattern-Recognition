"""Member 1 scratch inference. Official exports remain gated by shared M0.

The explicitly provisional width/scoring context is described in completion notes.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import time
import hashlib
import cv2
import numpy as np
import pandas as pd
from soccertrack import config, contracts, data_io, segmentation, proposals, patches, split as splitting


def rule_scores(proposals,cfg):
    """Provisional MWB-0 shape/fill score; optional fitted size agreement from propose.

    The private, named in-memory context is never serialized as a contract column.
    Without it, size agreement is neutral (the proposal's upstream filter applies).
    This is labeled provisional until the team approves the size-score interface.
    """
    if not len(proposals): return np.empty(0,dtype=np.float32)
    values=proposals[['w','h','fill_ratio']].to_numpy(float)
    if not np.isfinite(values).all() or (values[:,:2]<=1).any() or ((values[:,2]<0)|(values[:,2]>1)).any():
        raise ValueError('Invalid proposal dimensions/fill ratio')
    low,high=cfg['detector']['rules']['aspect_range']
    if not 0<low<high: raise ValueError('Invalid aspect range')
    aspect=values[:,1]/values[:,0]; center=(low+high)/2
    shape=np.clip(1-np.abs(aspect-center)/((high-low)/2),0,1)
    fill=np.clip(values[:,2]/cfg['detector']['rules']['min_fill_ratio'],0,1)
    agreement=np.asarray(proposals.attrs.get('member1_provisional_size_agreement',np.ones(len(proposals))),float)
    if agreement.shape!=(len(proposals),) or not np.isfinite(agreement).all(): raise ValueError('Invalid provisional size context')
    return np.clip(shape*fill*agreement,0,1).astype(np.float32)


def _nms(table,threshold):
    import torch
    from torchvision.ops import nms
    if not 0<=threshold<=1: raise ValueError('Invalid NMS threshold')
    if not len(table): return table.copy()
    boxes=table[['x','y','w','h']].to_numpy(np.float32,copy=True); boxes[:,2:]+=boxes[:,:2]
    indices=nms(torch.from_numpy(boxes),torch.tensor(table.score.to_numpy(np.float32)),threshold).numpy()
    return table.iloc[indices].sort_values(['frame','score'],ascending=[True,False]).reset_index(drop=True)


def prepare_frame_proposals(frame,frame_id,background,pitch,size_model,cfg):
    """Expose real intermediate masks/candidates for inspection without duplicating logic."""
    segmentation._image(frame)
    small=cv2.resize(frame,(background.shape[1],background.shape[0]),interpolation=cv2.INTER_AREA)
    raw_foreground=segmentation.foreground_mask(small,background,pitch,cfg)
    foreground=segmentation.clean_mask(raw_foreground,cfg)
    candidates=proposals.propose(foreground,frame_id,cfg['segmentation']['work_scale'],size_model,cfg)
    return candidates,raw_foreground,foreground


def classify_candidates(frame,candidates,cfg,classifier=None,score_floor=.05):
    """Score existing candidates and apply NMS, preserving the pre-NMS table."""
    if not 0<=score_floor<=1: raise ValueError('Score floor must be in [0,1]')
    candidates=candidates.copy()
    if classifier is None:
        scores=rule_scores(candidates,cfg)
    else:
        X=np.stack([patches.crop_patch(frame,tuple(row),cfg['patches']['context']) for row in candidates[['x','y','w','h']].to_numpy()]) if len(candidates) else np.empty((0,64,32,3),dtype=np.uint8)
        scores=classifier.predict_proba(X)
    scores=np.asarray(scores)
    if scores.shape!=(len(candidates),) or not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any():
        raise ValueError('Classifier must return one finite probability in [0,1] per proposal')
    candidates['score']=scores
    detections=candidates.loc[candidates.score>=score_floor,list(contracts.DETECTION_COLUMNS)].copy()
    detections=_nms(detections,cfg['detector']['nms_iou'])
    return candidates,detections


def detect_frame(frame,frame_id,background,pitch,size_model,cfg,classifier=None,score_floor=.05):
    """Member 1-owned in-memory inference; no exported/validated-artifact claim."""
    candidates,_,foreground=prepare_frame_proposals(frame,frame_id,background,pitch,size_model,cfg)
    candidates,detections=classify_candidates(frame,candidates,cfg,classifier,score_floor)
    return candidates,detections,foreground


def run_detection(detector_id,match_id,split,frameset,cfg,sources,frozen_test=False):
    splitting.require_test_access(split,frozen_test)
    if splitting.split_of(match_id,cfg)!=split: raise ValueError('Match/split mismatch')
    if detector_id not in ('scratch_mwb0','scratch_hog_svm','scratch_cnn'): raise ValueError('Unknown scratch detector')
    sc=cfg['scratch']; classifier=None
    if detector_id!='scratch_mwb0':
        from soccertrack.classifiers import load_classifier
        classifier=load_classifier(detector_id)
    # Shared provenance is required before any official artifact can be written.
    digest=config.config_sha256(); paths=config.match_paths(match_id,sources)
    info=data_io.video_info(paths.clip); total=data_io.count_decoded_frames(paths.clip)
    frames=config.frameset_range(frameset,total,cfg)
    train=[]
    for match in splitting.matches_for('train',cfg):
        if match in sources['matches']: train.append(data_io.load_gt_tracks(match,sources))
    if not train: raise ValueError('Training GT is required for the size model')
    training=pd.concat(train,ignore_index=True); training.attrs['split']='train'
    model=proposals.fit_size_model(training)
    bg=segmentation.estimate_background(paths.clip,sc['sampling']['background_num_frames'],total,sc['segmentation']['work_scale'])
    pitch=segmentation.pitch_mask(bg,sc,match_id,paths.calibration_dir)
    candidate_rows=[]; detected_rows=[]; processed=[]; start=time.perf_counter()
    for index,frame in data_io.iter_frames(paths.clip,frames):
        p,d,_=detect_frame(frame,index,bg,pitch,model,sc,classifier,cfg['evaluation']['score_floor'])
        candidate_rows.append(p); detected_rows.append(d); processed.append(index)
    if processed != list(frames): raise RuntimeError('Incomplete decoded coverage; no artifact published')
    seconds=time.perf_counter()-start
    det=pd.concat(detected_rows,ignore_index=True) if detected_rows else pd.DataFrame(columns=contracts.DETECTION_COLUMNS)
    prop=pd.concat(candidate_rows,ignore_index=True) if candidate_rows else pd.DataFrame(columns=contracts.PROPOSAL_COLUMNS)
    meta={'schema_version':contracts.SCHEMA_VERSION,'kind':'detections','method':detector_id,'tracker':None,
          'split':split,'match':match_id,'frameset':frameset,'frames':{'start':frames.start,'stop':frames.stop,'stride':frames.step},
          'source_num_frames':total,'processed_frame_count':len(processed),
          'image_width':info['width'],'image_height':info['height'],'fps':info['fps'],'score_floor':cfg['evaluation']['score_floor'],
          'operating_threshold':sc['detector']['operating_threshold'],'frozen_test':frozen_test,'config_sha256':digest,
          'git_commit':None,'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
          'runtime_seconds':seconds,'frames_per_second':len(detected_rows)/max(seconds,1e-9),'device':'cpu',
          'runtime_scope':'sequential decode/proposal/scoring/NMS; background/setup/write excluded',
          'notes':'Offline noncausal background; native coordinates; selected model/config publication required'}
    contracts.validate_detections(det,meta); contracts.validate_proposals(prop)
    contracts.validate_run_meta(meta)
    emitted=[]
    for kind,table in (('detections',det),('proposals',prop)):
        path=config.artifact_path(kind,detector_id,split,match_id,frameset)
        if path.exists() or contracts.meta_path(path).exists(): raise FileExistsError(path)
        contracts.write_table(table,path,kind)
        contracts.write_run_meta(dict(meta,kind=kind),path)
        emitted.extend([path,path.with_suffix('.meta.json')])
    segmentation.save_pitch_mask(pitch,match_id,(info['width'],info['height']))
    if split=='test':
        root=config.repo_root()
        hashes={}
        if classifier is not None:
            model_path=root/'outputs/models'/detector_id/('model.joblib' if detector_id=='scratch_hog_svm' else 'model.pt')
            hashes[model_path.relative_to(root).as_posix()]=hashlib.sha256(model_path.read_bytes()).hexdigest()
        splitting.log_test_access('Member 1',detector_id,[p.relative_to(root).as_posix() for p in emitted],hashes)
    return det
