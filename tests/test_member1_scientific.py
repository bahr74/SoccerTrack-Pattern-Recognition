"""Scientific-workflow invariants, actual pipeline composition and HTTP boundaries."""
import json,sys,threading
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demos/member1'))
import scientific_workflow as w
import app,experiment
from soccertrack import data_io,detector_scratch,segmentation,proposals

def test_clean_geometry_without_certification(fixtures_dir):
    raw=data_io.read_mot_raw(fixtures_dir/'mot_raw_dirty.txt')
    cleaned,issues=data_io.clean_mot_geometry(raw,(640,360),3)
    assert issues=={'duplicate_rows':1,'degenerate_boxes':1,'clipped_boxes':1,'frames_not_expected':1}
    assert len(cleaned)==11 and (cleaned.score==1).all()
    assert (cleaned.x>=0).all() and (cleaned.x+cleaned.w<=640).all()

def test_local_patch_baseline_and_confusion():
    result=w.patch_metrics(np.array([0,0,0,1]),np.zeros(4))
    assert result['accuracy']==.75 and result['f1']==0 and result['pr_auc_average_precision']==.25
    assert result['confusion_matrix']==[[3,0],[1,0]]

def test_no_duplicate_gt_matching():
    truth=pd.DataFrame([[0,1,10,10,20,40,1]],columns=['frame','track_id','x','y','w','h','score'])
    det=truth.drop(columns='track_id'); det=pd.concat([det,det],ignore_index=True)
    matched,fp,fn=w.match_frame(det,truth)
    assert len(matched)==1 and len(fp)==1 and not fn

def test_helper_composition_and_bad_probabilities():
    cfg=experiment.demo_config(); frame=np.full((180,320,3),(30,110,30),np.uint8)
    frame[40:90,100:120]=(40,40,170)
    bg=np.full((90,160,3),(30,110,30),np.uint8);pitch=np.ones((90,160),bool);size=proposals.SizeModel(0,50,.4)
    p,raw,clean=detector_scratch.prepare_frame_proposals(frame,0,bg,pitch,size,cfg)
    scored,det=detector_scratch.classify_candidates(frame,p,cfg)
    p2,det2,mask=detector_scratch.detect_frame(frame,0,bg,pitch,size,cfg)
    pd.testing.assert_frame_equal(scored,p2);pd.testing.assert_frame_equal(det,det2);np.testing.assert_array_equal(clean,mask)
    candidate=pd.DataFrame([[0,100,40,20,50,1000,.8,.5]],columns=['frame','x','y','w','h','area','fill_ratio','score'])
    class BadModel:
        def predict_proba(self,X): return np.full(len(X),float('nan'))
    with pytest.raises(ValueError):detector_scratch.classify_candidates(frame,candidate,cfg,BadModel())

def test_notebook_all_sections_and_clear_outputs():
    import nbformat
    p=Path(__file__).resolve().parents[1]/'demos/member1/notebooks/member1_scratch_pipeline_experiments.ipynb'
    book=nbformat.read(p,as_version=4)
    titles=[c.source.splitlines()[0] for c in book.cells if c.cell_type=='markdown' and c.source.startswith('## ')]
    assert titles==[f'## {i+1}. {title}' for i,title in enumerate(w.HEADINGS)]
    code=[c for c in book.cells if c.cell_type=='code']; assert len(code)==16
    assert all(c.execution_count is None and not c.outputs for c in code)
    assert any('RETRAIN = False' in c.source for c in code)

def test_upload_auth_file_decode_and_rejections(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'BASE',tmp_path/'isolated_output')
    source=tmp_path/'source.avi';experiment.make_synthetic(source,4)
    server=app.make_server(0);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def upload(name,body,token=True):
        headers={'X-Demo-Token':server.demo_token} if token else {}
        request=Request(base+'/api/upload?name='+name,body,headers=headers)
        with urlopen(request,timeout=20) as response:return json.load(response)
    try:
        value=upload('fixture.avi',source.read_bytes());assert Path(value['video']).is_relative_to(w.BASE)
        assert data_io.video_info(Path(value['video']))['frame_count_header']==4
        for name,body,token in [('128057.mp4',b'x',True),('..%2Fescape.mp4',b'x',True),('invalid.txt',b'x',True),('fixture.avi',source.read_bytes(),False)]:
            with pytest.raises(HTTPError):upload(name,body,token)
        with pytest.raises(HTTPError):upload('broken.mp4',b'not-a-video')
        assert not (tmp_path/'escape.mp4').exists()
    finally:server.shutdown();server.preview.close();server.server_close();worker.join(5)
