"""Member 1 online tracker acceptance tests."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack import tracker_iou as tracking,contracts

@pytest.fixture
def cfg():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/scratch.yaml').read_text())
    cfg['tracker_iou'].update(n_init=1,min_score=.5,max_age=1)
    return cfg

def test_01(fixtures_dir,cfg):
    gt=pd.read_csv(fixtures_dir/'tracks_gt.csv')
    tracks=tracking.track_sequence(gt.drop(columns='track_id'),range(3),cfg)
    assert len(tracks)==9
    for player in (1,2,3):
        actual=tracks.loc[tracks.x.isin(gt.loc[gt.track_id==player,'x'])]
        assert actual.track_id.nunique()==1

def test_02(cfg):
    rows=[]
    for f in range(10):
        rows += [[f,10+f*2,20,20,40,.9],[f,28-f*2,80,20,40,.9]]
    detections=pd.DataFrame(rows,columns=contracts.DETECTION_COLUMNS)
    tracks=tracking.track_sequence(detections,range(10),cfg)
    # Two distinct y lanes cross in x; no ambiguous identical-box occlusion claim.
    assert tracks.loc[tracks.y==20,'track_id'].nunique()==1
    assert tracks.loc[tracks.y==80,'track_id'].nunique()==1
    assert tracks.track_id.nunique()==2

def test_03():
    tracker=tracking.IouHungarianTracker(.3,3,1,.5)
    def det(f): return pd.DataFrame([[f,10,10,20,40,.9]],columns=contracts.DETECTION_COLUMNS)
    empty=pd.DataFrame(columns=contracts.DETECTION_COLUMNS)
    assert tracker.update(0,det(0)).empty and tracker.update(1,det(1)).empty
    assert tracker.update(2,det(2)).track_id.tolist()==[1]
    assert tracker.update(3,empty).empty
    assert tracker.update(4,empty).empty
    assert not tracker.tracks
    assert tracker.update(5,det(5)).empty

@pytest.mark.skip(reason='Member 1 M8 integration: shared contracts.validate_tracks is still Member 3 M0 placeholder')
def test_04(): ...

def test_empty_frames_and_validation(cfg):
    empty=pd.DataFrame(columns=contracts.DETECTION_COLUMNS)
    assert tracking.track_sequence(empty,range(4),cfg).empty
    with pytest.raises(ValueError,match='stride-1'): tracking.track_sequence(empty,range(0,4,2),cfg)
    tracker=tracking.IouHungarianTracker(.3,1,2,.5)
    tracker.update(0,empty)
    with pytest.raises(ValueError,match='consecutive'): tracker.update(2,empty)
    assert 'tracker_bytetrack' not in Path(tracking.__file__).read_text()

def test_iou_and_gated_assignment():
    a=np.array([[500,140,18,40],[0,0,10,10]])
    b=np.array([[510,150,18,40],[0,0,10,10]])
    result=tracking._iou(a,b)
    assert result[0,0]==pytest.approx(.2) and result[1,1]==1 and result[0,1]==0
    assert tracking._iou(np.empty((0,4)),b).shape==(0,2)
