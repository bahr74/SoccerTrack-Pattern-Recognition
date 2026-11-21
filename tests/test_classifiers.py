"""Member 1 bounded synthetic classifier acceptance and checkpoint tests."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pytest
import yaml
from soccertrack.classifiers import HogSvmClassifier,CnnClassifier

@pytest.fixture
def cfg():
    import torch
    torch.set_num_threads(1)
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/scratch.yaml').read_text())
    cfg['cnn']['device']='cpu'
    return cfg

def test_01(cfg,synthetic_patches):
    X,y=synthetic_patches()
    model=HogSvmClassifier(cfg).fit(X,y)
    score=model.predict_proba(X)
    assert ((score>=.5)==y).mean()>.95
    assert score.dtype==np.float32 and ((score>=0)&(score<=1)).all()

def test_02(cfg,synthetic_patches,tmp_path):
    X,y=synthetic_patches()
    model=HogSvmClassifier(cfg).fit_with_validation(X,y,*synthetic_patches(seed=2))
    path=tmp_path/'model.joblib'; model.save(path)
    loaded=HogSvmClassifier.load(path)
    np.testing.assert_array_equal(model.predict_proba(X),loaded.predict_proba(X))
    assert loaded.hog_weights().shape==(756,)

def test_03(cfg,synthetic_patches):
    model=CnnClassifier(cfg); X,_=synthetic_patches()
    assert sum(p.numel() for p in model.model.parameters())<100000
    assert model.model(model.preprocess(X)).shape==(32,1)
    with pytest.raises(RuntimeError,match='untrained'): model.predict_proba(X)

def test_04(cfg,synthetic_patches,tmp_path):
    cfg['cnn'].update(epochs=60,lr=.01,dropout=0,early_stopping_patience=60)
    cfg['cnn']['augment']={'hflip':False,'brightness':0,'contrast':0}
    X,y=synthetic_patches(n_pos=4,n_neg=4)
    model=CnnClassifier(cfg); history=model.fit(X,y,X,y)
    assert min(history['train_loss'])<.1
    assert history['val_pr_auc'][-1]>.95
    path=tmp_path/'model.pt'; model.save(path)
    loaded=CnnClassifier.load(path)
    np.testing.assert_allclose(model.predict_proba(X),loaded.predict_proba(X),atol=1e-6)

def test_05(cfg,synthetic_patches):
    import torch
    model=CnnClassifier(cfg); X,_=synthetic_patches()
    assert isinstance(model.gradcam_layer,torch.nn.Conv2d)
    assert model.preprocess(X).shape==(32,3,64,32)
    normalized=model.preprocess(X); normalized.requires_grad_(True)
    model.model(normalized).sum().backward()
    assert normalized.grad is not None
