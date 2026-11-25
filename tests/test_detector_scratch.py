"""Member 1 per-frame inference tests; official export remains an explicit dependency."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack import detector_scratch as detector, proposals, contracts

@pytest.fixture
def cfg():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/scratch.yaml').read_text())
    cfg['segmentation']['work_scale']=1.
    return cfg

def test_01(fixtures_dir,cfg):
    p=pd.read_csv(fixtures_dir/'proposals_sample.csv')
    scores=detector.rule_scores(p,cfg)
    assert scores.dtype==np.float32 and ((scores>=0)&(scores<=1)).all()
    assert scores[0]>scores[4]

def test_02(fixtures_dir,tmp_path):
    table=contracts.read_table(fixtures_dir/'detections_sample.csv','detections')
    path=contracts.write_table(table,tmp_path/'d.csv','detections')
    meta=contracts.read_run_meta(fixtures_dir/'detections_sample.csv')
    contracts.write_run_meta(meta,path)
    contracts.validate_detections(contracts.read_table(path,'detections'),contracts.read_run_meta(path))

def test_03(cfg):
    d=pd.DataFrame([[0,10,10,20,40,.8],[0,10,10,20,40,.9],[0,100,100,20,40,.7]],columns=contracts.DETECTION_COLUMNS)
    result=detector._nms(d,.5)
    assert result.score.tolist()==[.9,.7]

def test_synthetic_end_to_end_arrays(cfg,synthetic_frame,fixtures_dir):
    truth=pd.read_csv(fixtures_dir/'tracks_gt.csv').query('frame==0')
    boxes=truth[['x','y','w','h']].to_numpy()
    model=proposals.SizeModel(0,45,20/45)
    p,d,mask=detector.detect_frame(synthetic_frame(boxes),0,synthetic_frame(),np.ones((360,640),bool),model,cfg)
    assert len(p)==3 and len(d)==3
    assert tuple(d.columns)==contracts.DETECTION_COLUMNS and tuple(p.columns)==contracts.PROPOSAL_COLUMNS
    assert mask.dtype==bool and (d.score>=.05).all()
    assert contracts.validate_detections(d) is d
