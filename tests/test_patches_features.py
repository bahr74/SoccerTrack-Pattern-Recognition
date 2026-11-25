"""Member 1 patch, NumPy HOG and dataset-boundary acceptance tests."""
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack import patches,features,contracts
from soccertrack.tracker_iou import _iou

@pytest.fixture
def cfg():
    return yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/scratch.yaml').read_text())

def test_01(synthetic_frame):
    frame=synthetic_frame(pitch_bgr=(10,40,200))
    X=patches.crop_patch(frame,(100,100,20,40),context=0)
    assert X.shape==(64,32,3) and X.dtype==np.uint8
    np.testing.assert_array_equal(X[32,16],[200,40,10])
    border=patches.crop_patch(frame,(-10,-20,20,40),context=0)
    assert (border[:30,:15]==0).all()
    assert border[-1,-1,0]==200

def test_02(synthetic_frame,fixtures_dir,cfg):
    gt=pd.read_csv(fixtures_dir/'tracks_gt.csv').query('frame==0')
    candidates=pd.DataFrame({'x':[20.],'y':[20.],'w':[20.],'h':[45.]})
    rows=patches.sample_frame_patches(synthetic_frame(),gt,np.ones((360,640),bool),candidates,cfg,np.random.default_rng(42))
    assert len(rows)==12 and sum(r[1] for r in rows)==3
    expected=gt[['x','y','w','h']].to_numpy()
    for patch,label,box,kind in rows:
        assert patch.shape==(64,32,3)
        overlap=float(_iou(np.array([box]),expected).max())
        if kind=='partial': assert .1<=overlap<.4
        if kind in ('random','proposal'): assert overlap<.1
    repeated=patches.sample_frame_patches(synthetic_frame(),gt,np.ones((360,640),bool),candidates,cfg,np.random.default_rng(42))
    for a,b in zip(rows,repeated,strict=True): np.testing.assert_array_equal(a[2],b[2])

def test_03(tmp_path,cfg):
    with pytest.raises(PermissionError): patches.build_patch_set('test',{}, {},False)
    # Malformed arrays are rejected before any file is created.
    arrays={'X':np.zeros((1,64,32,3),np.uint8)}
    with pytest.raises(contracts.ContractError,match='missing'): patches.save_patch_set(arrays,tmp_path/'blocked.npz')
    assert not (tmp_path/'blocked.npz').exists()

def test_04(synthetic_patches,cfg):
    X,y=synthetic_patches()
    actual=features.hog_features(X,cfg)
    assert actual.shape==(32,756) and actual.dtype==np.float32 and np.isfinite(actual).all()
    assert features.hog_features(X[:0],cfg).shape==(0,756)
    assert features.hog_glyph(actual[0],cfg).shape==(64,32)
    hist=features.color_histograms(X,(8,4,4))
    np.testing.assert_allclose(hist.sum(axis=1),1)

def test_05(synthetic_patches,cfg):
    assert hasattr(cv2,'HOGDescriptor'), 'Reference parity BLOCKED: installed OpenCV 5.0 has no HOGDescriptor; dependency owner must provide an approved reference environment'
    X,_=synthetic_patches(n_pos=3,n_neg=3)
    own=features.hog_features(X,cfg).reshape(-1,7,3,2,2,9).transpose(0,2,1,4,3,5).reshape(-1,756)
    reference=cv2.HOGDescriptor((32,64),(16,16),(8,8),(8,8),9)
    ref=np.array([reference.compute(cv2.cvtColor(p,cv2.COLOR_RGB2GRAY)).ravel() for p in X])
    cosine=(own*ref).sum(axis=1)/(np.linalg.norm(own,axis=1)*np.linalg.norm(ref,axis=1))
    # No Gaussian block weighting; different spatial boundary treatment. Document
    # a meaningful orientation/structure tolerance rather than claiming equality.
    assert cosine.min()>.75
    zero=features.hog_features(np.zeros((1,64,32,3),np.uint8),cfg)
    assert not zero.any()

@pytest.mark.parametrize('box',[(0,0,-1,20),(0,0,float('nan'),20)])
def test_bad_crop(synthetic_frame,box):
    with pytest.raises(ValueError): patches.crop_patch(synthetic_frame(),box)
