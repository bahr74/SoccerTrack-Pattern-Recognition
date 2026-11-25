"""Independent ByteTrack adapter; top-left contract boxes, centre-based tracker input."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
from .contracts import DETECTION_COLUMNS, TRACK_COLUMNS

class DetectionsForByteTrack:
    def __init__(self,frame_dets: pd.DataFrame) -> None:
        self.conf=frame_dets.score.to_numpy(dtype=np.float32,copy=True)
        self.cls=np.zeros(len(frame_dets),np.float32)
        self.xywh=frame_dets[['x','y','w','h']].to_numpy(dtype=np.float32,copy=True)
        self.xywh[:,:2]+=self.xywh[:,2:]/2
    def __len__(self):
        return len(self.conf)
    def __getitem__(self,index):
        result=object.__new__(type(self))
        result.conf=np.atleast_1d(self.conf[index]).copy()
        result.cls=np.atleast_1d(self.cls[index]).copy()
        result.xywh=np.asarray(self.xywh[index]).reshape(-1,4).copy()
        return result

def track_sequence(detections: pd.DataFrame,frames: range,cfg: dict[str,Any]) -> pd.DataFrame:
    """Reset state per sequence; advance empty frames; retain low-confidence input.

    Kalman mode cannot clip to an unknown image extent; the artifact producer clips
    motion-filtered boxes to its metadata dimensions before validation/writing.
    """
    from .detector_yolo import _offline
    _offline()
    from ultralytics.trackers.byte_tracker import BYTETracker
    if not isinstance(frames,range) or frames.step!=1 or frames.start<0:
        raise ValueError('tracking requires a nonnegative contiguous range with stride 1')
    if tuple(detections.columns)!=DETECTION_COLUMNS:
        raise ValueError('invalid Detection columns/order')
    values=detections.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('detections must be finite')
    if len(values) and ((values[:,0]!=np.floor(values[:,0])).any() or (~detections.frame.isin(frames)).any()
        or (values[:,1:3]<0).any() or (values[:,3:5]<=1).any() or (values[:,5]<.05).any() or (values[:,5]>1).any()):
        raise ValueError('invalid frame membership, geometry or score')
    settings=cfg.get('yolo',cfg)['bytetrack'].copy()
    mode=settings.pop('output_box','detection')
    low,high,new=(settings[k] for k in ('track_low_thresh','track_high_thresh','new_track_thresh'))
    if mode not in ('detection','kalman') or not 0<=low<high<=1 or not high<=new<=1 or not 0<=settings['match_thresh']<=1:
        raise ValueError('invalid ByteTrack settings')
    if isinstance(settings['track_buffer'],bool) or not isinstance(settings['track_buffer'],int) or settings['track_buffer']<0:
        raise ValueError('invalid track_buffer')
    tracker=BYTETracker(SimpleNamespace(**settings))
    groups={int(f):g for f,g in detections.groupby('frame',sort=False)}
    rows=[]
    for frame in frames:
        group=groups.get(frame,detections.iloc[:0]).reset_index(drop=True)
        for row in tracker.update(DetectionsForByteTrack(group)):
            if len(row)!=8:
                raise RuntimeError('unsupported ByteTrack output, expected Nx8')
            index=int(row[7])
            if index!=row[7] or not 0<=index<len(group):
                raise RuntimeError('invalid source association index')
            if mode=='detection':
                x,y,w,h,score=group.iloc[index][['x','y','w','h','score']]
            else:
                x,y,x2,y2=row[:4]
                x,y=max(0.,float(x)),max(0.,float(y))
                w,h,score=x2-x,y2-y,row[5]
            if w>1 and h>1:
                rows.append((frame,int(row[4]),x,y,w,h,score))
    result=pd.DataFrame(rows,columns=TRACK_COLUMNS).astype({'frame':'int64','track_id':'int64'})
    return result.sort_values(['frame','track_id'],ignore_index=True)
