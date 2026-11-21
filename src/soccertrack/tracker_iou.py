"""Member 1 independent IoU/Hungarian tracker; no Pipeline B imports."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from soccertrack import contracts


def _iou(a,b):
    """Tracker-owned box geometry; does not replace Member 3's metrics interface."""
    a=np.asarray(a,dtype=float).reshape(-1,4); b=np.asarray(b,dtype=float).reshape(-1,4)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a[:,2:]<=0).any() or (b[:,2:]<=0).any():
        raise ValueError('IoU requires finite positive xywh boxes')
    top=np.maximum(a[:,None,:2],b[None,:,:2]); bottom=np.minimum(a[:,None,:2]+a[:,None,2:],b[None,:,:2]+b[None,:,2:])
    wh=np.maximum(bottom-top,0); inter=wh[:,:,0]*wh[:,:,1]
    return inter/np.maximum(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-inter,1e-12)


def _empty():
    return pd.DataFrame({c:pd.Series(dtype='int64' if c in ('frame','track_id') else 'float64') for c in contracts.TRACK_COLUMNS})


class IouHungarianTracker:
    def __init__(self,iou_gate,n_init,max_age,min_score):
        if not 0<=iou_gate<=1 or not 0<=min_score<=1 or not isinstance(n_init,int) or n_init<1 or not isinstance(max_age,int) or max_age<0:
            raise ValueError('Invalid tracker settings')
        self.iou_gate=iou_gate; self.n_init=n_init; self.max_age=max_age; self.min_score=min_score
        self.tracks={}; self.next_id=1; self.last_frame=None

    def update(self,frame_id,detections):
        if not isinstance(frame_id,(int,np.integer)) or frame_id<0 or (self.last_frame is not None and frame_id!=self.last_frame+1):
            raise ValueError('Tracker updates require consecutive 0-based frame indices')
        self.last_frame=int(frame_id)
        if len(detections):
            if not (detections.frame==frame_id).all(): raise ValueError('Detection rows must belong to the current frame')
            scores=detections.score.to_numpy(float)
            if not np.isfinite(scores).all() or ((scores<0)|(scores>1)).any(): raise ValueError('Scores must be in [0,1]')
            detections=detections.loc[detections.score>=self.min_score]
        boxes=detections[['x','y','w','h']].to_numpy(float)
        _iou(boxes,np.empty((0,4)))
        scores=detections.score.to_numpy(float)
        ids=list(self.tracks); used=set(); matched=set()
        if ids and len(boxes):
            overlap=_iou(np.array([self.tracks[i]['box'] for i in ids]),boxes)
            # Ineligible pairs must not consume an assignment that could satisfy the gate.
            cost=np.where(overlap>=self.iou_gate,1-overlap,1e6)
            rr,cc=linear_sum_assignment(cost)
            for r,c in zip(rr,cc):
                if overlap[r,c]<self.iou_gate: continue
                tid=ids[r]; track=self.tracks[tid]
                track.update(box=boxes[c].copy(),score=float(scores[c]),age=0,hits=track['hits']+1)
                track['confirmed']=track['confirmed'] or track['hits']>=self.n_init
                used.add(int(c)); matched.add(tid)
        for tid in ids:
            if tid not in matched:
                self.tracks[tid]['age']+=1; self.tracks[tid]['hits']=0
                if self.tracks[tid]['age']>self.max_age:
                    del self.tracks[tid]
        for c,box in enumerate(boxes):
            if c in used: continue
            tid=self.next_id; self.next_id+=1
            self.tracks[tid]={'box':box.copy(),'score':float(scores[c]),'hits':1,'age':0,'confirmed':self.n_init==1}
        rows=[]
        for tid,track in self.tracks.items():
            if track['confirmed'] and track['age']==0:
                rows.append([frame_id,tid,*track['box'],track['score']])
        return pd.DataFrame(rows,columns=contracts.TRACK_COLUMNS).astype({'frame':'int64','track_id':'int64'}) if rows else _empty()


def track_sequence(detections,frames,cfg):
    if not isinstance(frames,range) or frames.step!=1 or frames.start<0: raise ValueError('Tracking frames must be a stride-1 range')
    if len(detections) and not detections.frame.isin(frames).all(): raise ValueError('Detections lie outside frameset')
    sc=cfg.get('scratch',cfg); options=sc['tracker_iou']
    threshold=options['min_score']
    if threshold is None: threshold=sc['detector']['operating_threshold']
    if threshold is None: raise ValueError('Tracker needs an explicit detector operating threshold')
    tracker=IouHungarianTracker(options['iou_gate'],options['n_init'],options['max_age'],threshold)
    rows=[tracker.update(f,detections.loc[detections.frame==f]) for f in frames]
    result=pd.concat(rows,ignore_index=True) if rows else _empty()
    # No export/official validation claim is made here; exporters call shared contracts.
    return result
