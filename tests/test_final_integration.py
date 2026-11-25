"""Integrated infrastructure guard regressions; protected paths never resolved."""
import importlib.util
from pathlib import Path
import pytest
from soccertrack import config

@pytest.mark.parametrize('mid',['128057','132831','117092'])
def test_protected_match_before_resolution(mid):
    with pytest.raises(PermissionError):config.match_paths(mid,{'matches':{mid:object()}})

@pytest.mark.parametrize('bad',['../escape','C:/escape','/escape','..','a\\b'])
def test_artifact_component_containment(bad):
    with pytest.raises(ValueError):config.artifact_path('detections',bad,'val','117093','eval10')

def test_check_all_only_enabled_train_val():
    p=Path(__file__).resolve().parents[1]/'scripts/check_data.py'
    s=importlib.util.spec_from_file_location('safe_check',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    c=config.load_config()
    result=m._resolve_matches_for_split('all',{'matches':{'128057':{},'117092':{}}},c['project'])
    assert result==['118575','118576','118577','118578','128058','117093']
    with pytest.raises(PermissionError):m._resolve_matches_for_split('test',{},c['project'])
