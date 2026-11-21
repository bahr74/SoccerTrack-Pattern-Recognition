"""Member 1 isolated GUI/NPZ smoke checks; no shared-interface replacements."""
from pathlib import Path
import importlib.util
import sys
import json
import threading
from urllib.request import Request,urlopen
import cv2
import numpy as np
import pytest

DEMO=Path(__file__).resolve().parents[1]/'demos/member1'
sys.path.insert(0,str(DEMO))
import app
import experiment


def test_provisional_npz_schema(tmp_path):
    X=np.zeros((2,64,32,3),np.uint8); y=np.array([1,0],np.int8)
    records=[[0,10,10,20,40,'gt'],[0,100,100,20,40,'random']]
    path=tmp_path/'demo.npz'
    arrays=experiment.save_demo_patchset(X,y,records,'999001','train',path,42)
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files)=={'X','y','match','frame','box','kind','meta_json'}
        for key,value in arrays.items(): np.testing.assert_array_equal(saved[key],value)
        assert saved['frame'].dtype==np.int32 and saved['box'].dtype==np.float32
        assert saved['kind'].dtype==np.dtype('<U8')
        assert json.loads(saved['meta_json'][0])['status'].startswith('PROVISIONAL')


def test_gui_http_smoke_and_seek(tmp_path):
    source=tmp_path/'synthetic.avi'; experiment.make_synthetic(source,12)
    server=app.make_server(0); worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(endpoint,data):
        request=Request(base+'/api/'+endpoint,json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Demo-Token':server.demo_token})
        with urlopen(request,timeout=30) as response: return json.load(response)
    try:
        with urlopen(base) as response: page=response.read().decode()
        for text in ['Browse Video','Load Real Validation Demo','Video Analyzer','Pipeline Explorer','Model Comparison','Experiments &amp; Results','Previous frame','Next frame','Export annotated','type="file"']:
            assert text in page
        info=post('open',{'video':str(source)})
        assert info['preview_frames']==12
        first=post('frame',{'index':0,'mode':'rules','width':640})
        later=post('frame',{'index':5,'mode':'tracks','width':640})
        back=post('frame',{'index':1,'mode':'foreground','width':640})
        assert (first['frame'],later['frame'],back['frame'])==(0,5,1)
        assert (first['processed_frames'],later['processed_frames'],back['processed_frames'])==(1,6,1)
        cached=post('frame',{'index':1,'mode':'foreground','width':640})
        assert cached['cache_hit'] and cached['processing_fps'] is None
        exported=post('export',{'source':'synthetic','method':'rules'})
        assert exported['export_frames_decoded']==12 and exported['background_cache_hit']
        with urlopen(base+'/download?file='+exported['file']+'&token='+server.demo_token) as response:
            assert response.headers['Content-Type']=='video/mp4' and len(response.read())>100
        assert first['original'] and first['processed']
        from urllib.error import HTTPError
        with pytest.raises(HTTPError):
            urlopen(Request(base+'/api/frame',b'{}',headers={'Content-Type':'application/json'}))
        with pytest.raises(PermissionError): experiment.safe_video(tmp_path/'128057.mp4')
        with pytest.raises(HTTPError): post('open',{'video':str(tmp_path/'missing.avi')})
        with pytest.raises(HTTPError): post('frame',{'index':0})
    finally:
        server.shutdown(); server.preview.close(); server.server_close(); worker.join(timeout=5)


def test_provisional_diagnostic_matches_hand_fixture(fixtures_dir):
    import pandas as pd
    det=pd.read_csv(fixtures_dir/'detections_sample.csv'); gt=pd.read_csv(fixtures_dir/'tracks_gt.csv')
    result=experiment.local_detection_diagnostics(det,gt,range(3))
    assert (result['tp'],result['fp'],result['fn'])==(7,1,2)
    assert result['precision']==pytest.approx(.875)
    assert result['recall']==pytest.approx(7/9)


def test_experiment_pair_guards():
    cfg=experiment.demo_config()
    experiment.validate_experiment_pairs('118575.mp4','118575.txt','117093.mp4','117093.txt',cfg)
    with pytest.raises(ValueError,match='must agree'):
        experiment.validate_experiment_pairs('118575.mp4','117093.txt','117093.mp4','118575.txt',cfg)
    with pytest.raises(PermissionError):
        experiment.validate_experiment_pairs('128057.mp4','128057.txt','117093.mp4','117093.txt',cfg)
