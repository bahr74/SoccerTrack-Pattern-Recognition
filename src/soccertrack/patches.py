"""Member 1 RGB patch crops and bounded, reproducible train/validation sampling."""
from __future__ import annotations
from pathlib import Path
import json
import cv2
import numpy as np
from soccertrack import config, contracts, data_io, split as splitting, segmentation, proposals


def crop_patch(frame_bgr, box, context=.10, size=(64,32)):
    segmentation._image(frame_bgr)
    values=np.asarray(box,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all() or min(values[2:])<=0 or not np.isfinite(context) or not 0<=context<=1:
        raise ValueError('Expected finite xywh with positive size and context in [0,1]')
    if len(size)!=2 or any(not isinstance(i,(int,np.integer)) or i<1 for i in size):
        raise ValueError('size must be positive integer (height,width)')
    x,y,w,h=values; left=int(np.floor(x-context*w)); top=int(np.floor(y-context*h))
    right=int(np.ceil(x+w+context*w)); bottom=int(np.ceil(y+h+context*h))
    if (right-left)*(bottom-top)>16*frame_bgr.shape[0]*frame_bgr.shape[1]:
        raise ValueError('Crop extent is excessively larger than the frame')
    padded=np.zeros((bottom-top,right-left,3),dtype=np.uint8)
    sx,sy=max(left,0),max(top,0); ex,ey=min(right,frame_bgr.shape[1]),min(bottom,frame_bgr.shape[0])
    if ex>sx and ey>sy: padded[sy-top:ey-top,sx-left:ex-left]=frame_bgr[sy:ey,sx:ex]
    return cv2.cvtColor(cv2.resize(padded,(size[1],size[0]),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)


def sample_frame_patches(frame, gt, pitch, candidates, cfg, rng):
    """Member 1-owned pure sampler; return (patch, label, xywh, kind) records.

    Finite rejection sampling fails explicitly when required negative pools cannot
    supply the configured mixture. It never relabels a negative or changes ratios.
    """
    from soccertrack.tracker_iou import _iou
    segmentation._image(frame); segmentation._mask(pitch)
    if pitch.shape!=frame.shape[:2]: raise ValueError('Pitch/frame resolutions differ')
    boxes=gt[['x','y','w','h']].to_numpy(float); options=cfg['patches']
    if not len(boxes): return []
    records=[(crop_patch(frame,tuple(b),options['context'],(options['height'],options['width'])),1,b.copy(),'gt') for b in boxes]
    count=int(len(boxes)*options['negatives_per_positive'])
    kinds=['random','partial','proposal']; mix=np.array([options['negative_mix'][k] for k in kinds],float)
    if (mix<0).any() or not np.isclose(mix.sum(),1): raise ValueError('Negative mixture must sum to 1')
    quotas=np.floor(mix*count).astype(int)
    for i in np.argsort(-(mix*count-quotas),kind='stable')[:count-quotas.sum()]: quotas[i]+=1
    candidate_boxes=candidates[['x','y','w','h']].to_numpy(float)
    H,W=frame.shape[:2]
    for kind,needed in zip(kinds,quotas):
        found=0
        for attempt in range(max(1000,int(needed)*200)):
            if found>=needed: break
            base=boxes[int(rng.integers(len(boxes)))]; w,h=base[2:]
            if kind=='proposal':
                if not len(candidate_boxes): break
                box=candidate_boxes[int(rng.integers(len(candidate_boxes)))].copy()
            elif kind=='partial':
                box=base.copy(); box[0]+=rng.uniform(-1,1)*w; box[1]+=rng.uniform(-.5,.5)*h
            else:
                if w>W or h>H: continue
                box=np.array([rng.uniform(0,W-w),rng.uniform(0,H-h),w,h])
            x,y,w,h=box
            if min(x,y)<0 or x+w>W or y+h>H or min(w,h)<=1: continue
            overlap=float(_iou(np.array([box]),boxes).max())
            if kind=='partial':
                lo,hi=options['partial_iou_range']
                valid=lo<=overlap<hi
            else:
                valid=overlap<options['random_iou_max']
            if not valid: continue
            if kind=='random' and not pitch[int(y):int(np.ceil(y+h)),int(x):int(np.ceil(x+w))].all(): continue
            records.append((crop_patch(frame,tuple(box),options['context'],(options['height'],options['width'])),0,box,kind)); found+=1
        if found<needed: raise ValueError(f'Insufficient {kind} negatives: {found}/{needed}; no ratio fallback')
    return records


def build_patch_set(split, cfg, sources, frozen_test=False):
    splitting.require_test_access(split,frozen_test)
    project=cfg['project']; sc=cfg['scratch']; rng=np.random.default_rng(project['seed'])
    matches=splitting.matches_for(split,cfg)
    train=[]
    for match in splitting.matches_for('train',cfg):
        if match in sources['matches']: train.append(data_io.load_gt_tracks(match,sources))
    import pandas as pd
    if not train: raise ValueError('Size model requires available training GT')
    training=pd.concat(train,ignore_index=True); training.attrs['split']='train'
    size_model=proposals.fit_size_model(training)
    records=[]; match_ids=[]; frames=[]
    for match in matches:
        if match not in sources['matches']: continue
        paths=config.match_paths(match,sources); gt=data_io.load_gt_tracks(match,sources)
        total=data_io.count_decoded_frames(paths.clip); scale=sc['segmentation']['work_scale']
        bg=segmentation.estimate_background(paths.clip,sc['sampling']['background_num_frames'],total,scale)
        mask=segmentation.pitch_mask(bg,sc,match,paths.calibration_dir)
        for index,frame in data_io.iter_frames(paths.clip,range(0,total,sc['sampling']['train_frame_stride'])):
            small=cv2.resize(frame,(bg.shape[1],bg.shape[0]),interpolation=cv2.INTER_AREA)
            fg=segmentation.clean_mask(segmentation.foreground_mask(small,bg,mask,sc),sc)
            candidates=proposals.propose(fg,index,scale,size_model,sc)
            fullmask=cv2.resize(mask.astype(np.uint8),(frame.shape[1],frame.shape[0]),interpolation=cv2.INTER_NEAREST).astype(bool)
            rows=sample_frame_patches(frame,gt.loc[gt.frame==index],fullmask,candidates,sc,rng)
            records.extend(rows); match_ids.extend([match]*len(rows)); frames.extend([index]*len(rows))
    if not records: raise ValueError('No patches sampled')
    arrays={'X':np.stack([r[0] for r in records]),'y':np.array([r[1] for r in records],dtype=np.int8),
            'match':np.array(match_ids,dtype='<U6'),'frame':np.array(frames,dtype=np.int32),
            'box':np.array([r[2] for r in records],dtype=np.float32),'kind':np.array([r[3] for r in records],dtype='<U8'),
            'meta_json':np.array([json.dumps({'schema_version':contracts.SCHEMA_VERSION,'split':split,'ratio':sc['patches']['negatives_per_positive'],
                'seed':project['seed'],'sampling_stride':sc['sampling']['train_frame_stride'],'iou_bounds':{'random_max':sc['patches']['random_iou_max'],'partial':sc['patches']['partial_iou_range']}})])}
    contracts.validate_patch_set(arrays)
    return arrays


def save_patch_set(arrays,path):
    contracts.validate_patch_set(arrays)
    path=Path(path)
    if path.suffix!='.npz': raise ValueError('Patch set path must end in .npz')
    path.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(path,**arrays)
    return path


def load_patch_set(path):
    with np.load(path,allow_pickle=False) as archive: arrays={key:archive[key] for key in archive.files}
    contracts.validate_patch_set(arrays)
    return arrays
