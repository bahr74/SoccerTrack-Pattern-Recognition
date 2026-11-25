"""Tiling, mocked inference and official-boundary tests; real smoke is explicit."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack import config,contracts,data_io
from soccertrack.detector_yolo import make_tiles,tile_to_frame,merge_tile_detections,apply_pitch_filter,YoloTiledDetector,run_detection

ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def cfg():
    return {'yolo':yaml.safe_load((ROOT/'configs/yolo.yaml').read_text(encoding='utf-8-sig')),
            'evaluation':yaml.safe_load((ROOT/'configs/evaluation.yaml').read_text()),
            'project':yaml.safe_load((ROOT/'configs/project.yaml').read_text())}

@pytest.mark.parametrize('size',[(4096,1080),(3840,1504),(400,200),(1280,1080),(1,1)])
def test_coverage(size):
    width,height=size
    tiles=make_tiles(width,height,1280,1080,64)
    coverage=np.zeros((height,width),bool)
    assert len(tiles)==len(set(tiles))
    for x0,y0,x1,y1 in tiles:
        assert 0<=x0<x1<=width and 0<=y0<y1<=height
        coverage[y0:y1,x0:x1]=True
    assert coverage.all()
    assert max(t[2] for t in tiles)==width and max(t[3] for t in tiles)==height
    starts=sorted(set(t[0] for t in tiles))
    assert all(a+1280-b>=64 for a,b in zip(starts,starts[1:]))

@pytest.mark.parametrize('args',[(0,10,5,5,1),(10,10,0,5,0),(10,10,5,5,5),(10,10,5,5,-1),(True,10,5,5,1),(10,10,5,5,.5)])
def test_invalid_tiling(args):
    with pytest.raises(ValueError): make_tiles(*args)

def test_translation_roundtrip_and_no_mutation():
    boxes=np.array([[2,3,22,43]],np.float32)
    np.testing.assert_array_equal(tile_to_frame(boxes,(1216,0,2496,1080)),[[1218,3,1238,43]])
    np.testing.assert_array_equal(boxes,[[2,3,22,43]])
    assert tile_to_frame(np.empty((0,4)),(0,0,2,2)).shape==(0,4)

def test_nms_duplicates_empty_invalid():
    boxes=np.array([[100,50,120,95],[101,50,121,95],[300,50,320,95],[0,0,0,5],[np.nan,0,5,5]])
    kept,scores=merge_tile_detections(boxes,np.array([.9,.8,.7,.6,.5]),.5)
    np.testing.assert_array_equal(kept,boxes[[0,2]])
    np.testing.assert_allclose(scores,[.9,.7])
    assert merge_tile_detections(np.empty((0,4)),np.empty(0),.5)[0].shape==(0,4)
    with pytest.raises(ValueError): merge_tile_detections(boxes,np.empty(0),.5)

def test_pitch_missing_and_footpoint(tmp_path):
    import cv2
    table=pd.DataFrame([[0,0,0,4,4,.5],[0,6,0,3,4,.8]],columns=contracts.DETECTION_COLUMNS)
    assert apply_pitch_filter(table,tmp_path/'absent.png')[0] is table
    mask=np.zeros((10,10),np.uint8); mask[4,2]=255
    cv2.imwrite(str(tmp_path/'mask.png'),mask)
    table.attrs['image_size']=(10,10)
    result,note=apply_pitch_filter(table,tmp_path/'mask.png')
    assert len(result)==1 and note=='pitch_filter=on'
    table.attrs['image_size']=(20,10)
    with pytest.raises(ValueError): apply_pitch_filter(table,tmp_path/'mask.png')

def mock_yolo(monkeypatch):
    import torch
    import soccertrack.detector_yolo as module
    module._offline()
    import ultralytics
    calls=[]
    class Fake:
        def __init__(self,path,task): calls.append((path,task))
        def predict(self,crops,**kwargs):
            calls.append(kwargs)
            result=[]
            for crop in crops:
                result.append(SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor([[2,3,22,43],[-5,0,8,9],[1,1,3,3]]),
                    conf=torch.tensor([.1,.8,.9]),cls=torch.tensor([0,0,1]))))
            return result
    monkeypatch.setattr(ultralytics,'YOLO',Fake)
    return calls

def test_detector_mock_offsets_classes_floor_clipping(monkeypatch,tmp_path,cfg):
    calls=mock_yolo(monkeypatch)
    weights=tmp_path/'weights.pt'; weights.write_bytes(b'mocked')
    detector=YoloTiledDetector(cfg,weights)
    result=detector.detect_frame(np.zeros((1080,4096,3),np.uint8),7)
    assert tuple(result.columns)==contracts.DETECTION_COLUMNS
    assert len(result)==8 and set(result.frame)=={7} and result.score.min()>=.05
    assert (result.x>=0).all() and ((result.x+result.w)<=4096).all()
    assert cfg['yolo']['tiling']['tile_width']-cfg['yolo']['tiling']['overlap_px']+2 in result.x.to_list()
    assert calls[-1]['classes']==[0] and calls[-1]['conf']==.05 and calls[-1]['save'] is False
    with pytest.raises(ValueError): detector.detect_frame(np.zeros((5,5,3),float),0)
    with pytest.raises(FileNotFoundError): YoloTiledDetector(cfg,'missing.pt')

def test_official_runner_mocked_shared_boundary(monkeypatch,tmp_path,cfg):
    mock_yolo(monkeypatch)
    weights=tmp_path/'weights.pt'; weights.write_bytes(b'mocked')
    monkeypatch.setattr(config,'match_paths',lambda *a:SimpleNamespace(clip=tmp_path/'clip.avi'))
    monkeypatch.setattr(config,'weights_path',lambda *a:weights)
    monkeypatch.setattr(config,'frameset_range',lambda *a:range(3))
    monkeypatch.setattr(config,'artifact_path',lambda *a:tmp_path/'output.csv')
    monkeypatch.setattr(config,'set_seeds',lambda *a:None)
    monkeypatch.setattr(config,'config_sha256',lambda:'a'*64)
    monkeypatch.setattr(config,'repo_root',lambda:ROOT)
    monkeypatch.setattr(data_io,'video_info',lambda *a:dict(width=40,height=50,fps=25.,frame_count_header=3))
    monkeypatch.setattr(data_io,'iter_frames',lambda *a:((i,np.zeros((50,40,3),np.uint8)) for i in range(3)))
    captured={}
    def validate(table,meta):
        assert tuple(table.columns)==contracts.DETECTION_COLUMNS
        assert set(table.frame)=={0,1,2}
        assert meta['frames']==dict(start=0,stop=3,stride=1)
        assert meta['created_utc'].endswith('Z')
        assert 'weights_sha256=' in meta['notes']
        captured['meta']=meta
        return table
    monkeypatch.setattr(contracts,'validate_detections',validate)
    monkeypatch.setattr(contracts,'validate_run_meta',lambda m:m)
    monkeypatch.setattr(contracts,'write_table',lambda df,p,k:captured.update(table=df.copy()))
    monkeypatch.setattr(contracts,'write_run_meta',lambda m,p:None)
    result=run_detection('yolov8n_coco','117093','val','smoke',cfg,{})
    assert len(result)==6 and 'table' in captured
    with pytest.raises(PermissionError): run_detection('yolov8n_coco','128057','test','smoke',cfg,{})

@pytest.mark.gpu
@pytest.mark.data
def test_real_validation_smoke(cfg):
    weights=Path('F:/Football-Fatigue-Project/models/yolov8n.pt')
    clip=Path('F:/Football-Fatigue-Project/data/external/soccertrack-v2/raw/117093/117093.mp4')
    import torch
    if not weights.is_file() or not clip.is_file() or not torch.cuda.is_available():
        pytest.skip('requires approved local validation clip/weights and CUDA')
    generator=data_io.iter_frames(clip,[0])
    try: frame,image=next(generator)
    finally: generator.close()
    detector=YoloTiledDetector(cfg,weights)
    assert detector.weights_sha256=='f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'
    result=detector.detect_frame(image,frame)
    assert len(result)>10
    assert ((result.x>=0)&(result.y>=0)&(result.w>1)&(result.h>1)&(result.x+result.w<=image.shape[1])&(result.y+result.h<=image.shape[0])).all()

@pytest.mark.parametrize('width,height,tile_w,tile_h,overlap',[(4096,1080,1280,1080,64),(3840,1504,1280,1080,128),(320,180,1280,1080,64),(1280,1080,1280,1080,0),(4096,1080,960,1080,64)])
def test_exact_tile_coverage(width,height,tile_w,tile_h,overlap):
    from soccertrack.detector_yolo import make_tiles
    tiles=make_tiles(width,height,tile_w,tile_h,overlap)
    coverage=np.zeros((height,width),bool)
    assert len(tiles)==len(set(tiles))
    for x0,y0,x1,y1 in tiles:
        assert 0<=x0<x1<=width and 0<=y0<y1<=height
        coverage[y0:y1,x0:x1]=True
    assert coverage.all()
    assert max(t[2] for t in tiles)==width and max(t[3] for t in tiles)==height
    for axis,size,extent in [(0,tile_w,width),(1,tile_h,height)]:
        starts=sorted(set(t[axis] for t in tiles))
        for left,right in zip(starts,starts[1:]):
            assert left+size-right>=overlap

@pytest.mark.parametrize('args',[(0,100,10,10,1),(100,0,10,10,1),(100,100,0,10,1),(100,100,10,10,-1),(100,100,10,10,10),(True,100,10,10,1),(100.,100,10,10,1)])
def test_invalid_tiling(args):
    from soccertrack.detector_yolo import make_tiles
    with pytest.raises(ValueError): make_tiles(*args)


def test_incomplete_decode_releases_generator_and_does_not_write(monkeypatch,tmp_path,cfg):
    from soccertrack import config,contracts,data_io
    import soccertrack.detector_yolo as module
    closed=[]
    published=[]
    class FakeDetector:
        last_timing={'processing_seconds':.01}
        def __init__(self,*a): pass
        def detect_frame(self,image,frame):
            return pd.DataFrame(columns=contracts.DETECTION_COLUMNS)
    def frames(*args):
        try:
            yield 0,np.zeros((10,10,3),np.uint8)
        finally:
            closed.append(True)
    monkeypatch.setattr(module,'YoloTiledDetector',FakeDetector)
    monkeypatch.setattr(config,'match_paths',lambda *a:SimpleNamespace(clip=tmp_path/'none'))
    monkeypatch.setattr(config,'weights_path',lambda *a:tmp_path/'none')
    monkeypatch.setattr(config,'frameset_range',lambda *a:range(3))
    monkeypatch.setattr(config,'artifact_path',lambda *a:tmp_path/'output.csv')
    monkeypatch.setattr(config,'set_seeds',lambda *a:None)
    monkeypatch.setattr(data_io,'video_info',lambda *a:dict(width=10,height=10,fps=25.,frame_count_header=3))
    monkeypatch.setattr(data_io,'iter_frames',frames)
    monkeypatch.setattr(contracts,'write_table',lambda *a:published.append(True))
    with pytest.raises(RuntimeError,match='coverage'):
        module.run_detection('yolov8n_coco','117093','val','smoke',cfg,{})
    assert closed==[True] and published==[] and not (tmp_path/'output.csv').exists()
