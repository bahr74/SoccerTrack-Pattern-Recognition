"""Actual installed ByteTrack integration on deterministic synthetic observations."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack.contracts import DETECTION_COLUMNS,TRACK_COLUMNS
from soccertrack.tracker_bytetrack import DetectionsForByteTrack,track_sequence
@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/yolo.yaml').read_text(encoding='utf-8-sig'))

def sequence(frames=range(8),objects=1):
    return pd.DataFrame([(f,10+f*2+j*100,20,20,40,.9) for f in frames for j in range(objects)],columns=DETECTION_COLUMNS)

def test_adapter():
    table=sequence(range(1),2)
    adapter=DetectionsForByteTrack(table)
    np.testing.assert_allclose(adapter.xywh,[[20,40,20,40],[120,40,20,40]])
    assert len(adapter[np.array([True,False])])==1
    assert len(adapter[[1,0]])==2 and len(adapter[0])==1
    assert adapter[np.array([False,False])].xywh.shape==(0,4)
    assert np.all(adapter.cls==0)

@pytest.mark.parametrize('objects',[1,3])
def test_continuity_and_reset(cfg,objects):
    dets=sequence(objects=objects)
    a=track_sequence(dets,range(8),cfg)
    b=track_sequence(dets,range(8),cfg)
    pd.testing.assert_frame_equal(a,b)
    assert len(a)==8*objects and a.track_id.nunique()==objects
    assert tuple(a.columns)==TRACK_COLUMNS
    assert (a.track_id>0).all()
    np.testing.assert_allclose(a[['x','y','w','h','score']],dets[['x','y','w','h','score']])

def test_empty_and_missed_frames(cfg):
    empty=sequence(range(0))
    assert track_sequence(empty,range(0),cfg).empty
    assert track_sequence(empty,range(4),cfg).empty
    dets=sequence().query('frame != 3').reset_index(drop=True)
    tracks=track_sequence(dets,range(8),cfg)
    assert 3 not in tracks.frame.to_list() and tracks.track_id.nunique()==1
    assert len(tracks)==7

def test_low_score_full_input_index_mapping(cfg):
    dets=sequence(range(3),2)
    # Low-score row comes first; high subset indices differ from full-input indices.
    dets.loc[(dets.frame==1)&(dets.x<100),'score']=.15
    tracks=track_sequence(dets,range(3),cfg)
    assert len(tracks)==6 and tracks.track_id.nunique()==2
    low=tracks[(tracks.frame==1)&(tracks.x<100)].iloc[0]
    assert low.score==pytest.approx(.15) and low.x==12

def test_late_initialization_confirmation(cfg):
    tracks=track_sequence(sequence(range(2,6)),range(6),cfg)
    assert tracks.frame.min()==3 and tracks.track_id.nunique()==1

def test_removal_after_buffer(cfg):
    cfg['bytetrack']['track_buffer']=1
    dets=pd.concat([sequence(range(2)),sequence(range(6,10))],ignore_index=True)
    tracks=track_sequence(dets,range(10),cfg)
    assert tracks.track_id.nunique()==2

def test_detection_vs_kalman(cfg):
    dets=sequence()
    dets.loc[dets.frame==4,'x']+=7
    raw=track_sequence(dets,range(8),cfg)
    cfg['bytetrack']['output_box']='kalman'
    filtered=track_sequence(dets,range(8),cfg)
    assert raw[raw.frame==4].iloc[0].x==25
    assert filtered[filtered.frame==4].iloc[0].x!=pytest.approx(25,abs=.01)
    assert (filtered[['w','h']]>1).all().all()

def test_crossing_is_valid_not_identity_guarantee(cfg):
    rows=[]
    for f in range(15):
        rows.extend([(f,50+f*3,20,20,40,.9),(f,140-f*3,22,20,40,.9)])
    tracks=track_sequence(pd.DataFrame(rows,columns=DETECTION_COLUMNS),range(15),cfg)
    assert not tracks.duplicated(['frame','track_id']).any()
    assert (tracks.track_id>0).all()
    assert len(tracks)>20

@pytest.mark.parametrize('frames',[range(0,8,2),range(-1,8),[0,1,2]])
def test_reject_sparse_or_invalid_frames(cfg,frames):
    with pytest.raises(ValueError): track_sequence(sequence(),frames,cfg)

def test_reject_invalid_detection_and_config(cfg):
    bad=sequence(); bad.loc[0,'score']=float('nan')
    with pytest.raises(ValueError): track_sequence(bad,range(8),cfg)
    cfg['bytetrack']['track_low_thresh']=.9
    with pytest.raises(ValueError): track_sequence(sequence(),range(8),cfg)


def test_cli_phase3_and_sparse_tracking_guards(monkeypatch,tmp_path):
    import scripts.run_yolo as cli
    import sys
    from soccertrack import config,contracts
    bundle={name:yaml.safe_load((Path(__file__).resolve().parents[1]/f'configs/{name}.yaml').read_text(encoding='utf-8-sig')) for name in ['project','yolo','evaluation']}
    monkeypatch.setattr(sys,'argv',['run_yolo.py','finetune'])
    with pytest.raises(SystemExit,match='Phase 3'):
        cli.main()
    monkeypatch.setattr(sys,'argv',['run_yolo.py','track','--match','117093','--frameset','eval10'])
    monkeypatch.setattr(config,'load_config',lambda:bundle)
    monkeypatch.setattr(config,'load_data_sources',lambda:{})
    monkeypatch.setattr(config,'set_seeds',lambda *a:None)
    monkeypatch.setattr(config,'artifact_path',lambda *a:tmp_path/'fixture.csv')
    monkeypatch.setattr(contracts,'read_run_meta',lambda *a:dict(method='yolov8n_coco',match='117093',split='val',frameset='eval10',frames=dict(start=0,stop=100,stride=10)))
    monkeypatch.setattr(contracts,'read_table',lambda *a:pytest.fail('sparse tracking must fail before loading detections'))
    with pytest.raises(ValueError,match='stride 1'):
        cli.main()
