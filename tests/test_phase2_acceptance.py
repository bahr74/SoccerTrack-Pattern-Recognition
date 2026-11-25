"""Real accepted-artifact/API verification, explicitly selected with -m data."""
from pathlib import Path
import base64,json,sys
import cv2
import numpy as np
import pytest
from soccertrack import contracts as C,config,detector_scratch,classifiers,tracker_iou,tracker_bytetrack
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/phase2_final/phase2_20261125_01'
pytestmark=pytest.mark.data

def test_accepted_tables_and_counts():
    d=json.loads((RUN/'accepted_results.json').read_text())
    assert d['config_sha256']==config.config_sha256()
    for item in d['artifacts'].values():
        path=RUN/item['path'];meta=C.read_run_meta(path)
        table=C.read_table(path,meta['kind'])
        C.validate_run_meta(meta)
        assert set(table.frame)<=set(range(meta['frames']['start'],meta['frames']['stop'],meta['frames']['stride']))
    for doc in d['results']:
        C.validate_metrics(doc)
        if doc['eval_id']=='E3':assert doc['counts']['frames']==598 and doc['counts']['gt']==13156
        if doc['eval_id']=='E4':assert doc['counts']['frames']==1500 and doc['counts']['gt']==33000

def test_cache_only_provider_original_gt_seek_and_overlay(monkeypatch):
    sys.path.insert(0,str(ROOT/'demos/member1'))
    from final_provider import FinalProvider
    def forbidden(*a,**k):pytest.fail('Cache/original/GT view called inference or association')
    monkeypatch.setattr(detector_scratch,'detect_frame',forbidden)
    monkeypatch.setattr(classifiers,'load_classifier',forbidden)
    monkeypatch.setattr(tracker_iou,'track_sequence',forbidden)
    monkeypatch.setattr(tracker_bytetrack,'track_sequence',forbidden)
    p=FinalProvider(RUN)
    try:
        a=p.frame(7,view='original');b=p.frame(7,view='gt')
        assert a['original']==a['processed']==b['original'] and b['processed']!=b['original']
        shape=cv2.imdecode(np.frombuffer(base64.b64decode(a['original']),np.uint8),cv2.IMREAD_COLOR).shape
        assert shape[:2]==(338,1280)
        for f in (0,10,1,1499,1500):
            d=p.frame(f,'yolov8n_coco','detection')
            assert d['frame']==f and d['scope']=='cached canonical predictions'
        d=p.frame(0,'yolov8n_coco','tracks');assert d['counts']['visible_ids']>=0
        with pytest.raises(ValueError,match='No cached'):p.frame(1501,'yolov8n_coco','detection')
        with pytest.raises(ValueError,match='1500|1499'):p.frame(1500,'yolov8n_coco','tracks')
    finally:p.close()

def test_executed_notebook_and_video():
    import nbformat
    nb=nbformat.read(RUN/'phase2_baseline_executed.ipynb',as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    assert len(code)>=15 and all(c.execution_count is not None for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    video=json.loads((RUN/'video_verification.json').read_text())
    assert video['frames']==150 and video['fps']==25 and video['source_frames']==list(range(150))
