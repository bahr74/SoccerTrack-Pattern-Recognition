"""Member1/2 export bridges tested with synthetic fixtures and no production config replacements."""
import importlib.util
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
import yaml
from soccertrack import contracts as C, evaluate as E, patches
from demos.member2 import safety, migrate_cache as migration
from demos.member2.audit import digest

ROOT=Path(__file__).resolve().parents[1]

def scratch_runner():
    spec=importlib.util.spec_from_file_location('scratch_export', ROOT/'scripts/run_scratch.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def arrays():
    return dict(X=np.zeros((2,64,32,3),np.uint8),y=np.array([1,0],np.int8),
                match=np.array(['117093','117093']),frame=np.array([0,0],np.int32),
                box=np.array([[0,0,2,2],[3,0,2,2]],np.float32),kind=np.array(['gt','random']),
                meta_json=np.array([json.dumps(dict(schema_version='1.0',split='val'))]))

def test_scratch_score_export_schema_roundtrip_and_no_overwrite(tmp_path,monkeypatch):
    from soccertrack import config
    monkeypatch.setattr(config,'config_sha256',lambda:'b'*64)
    runner=scratch_runner(); a=arrays();cfg=E.load_evaluation_config()
    cfg['scratch']=yaml.safe_load((ROOT/'configs/scratch.yaml').read_text())
    class Model:
        def predict_proba(self,X): return np.array([.876543,.123456])
    p=runner.export_patch_scores(a,Model(),'scratch_hog_svm','val',cfg,tmp_path/'scores.csv','a'*64,'b'*64)
    d=C.read_table(p,'patch_scores');m=C.read_run_meta(p)
    assert d.score.tolist()==[.8765,.1235]
    assert all(m[k] is None for k in ('match','frames','frameset','fps','image_width','image_height'))
    assert m['model_sha256']=='a'*64 and m['created_utc'].endswith('Z')
    assert E.evaluate_patches(p,cfg)['counts']==dict(tp=1,fp=0,tn=1,fn=0)
    with pytest.raises(FileExistsError): runner.export_patch_scores(a,Model(),'scratch_hog_svm','val',cfg,p,'a'*64,'b'*64)
    a['match'][0]='128057'
    with pytest.raises(PermissionError): runner.export_patch_scores(a,Model(),'scratch_hog_svm','val',cfg,tmp_path/'denied.csv','a'*64,'b'*64)
    assert not (tmp_path/'denied.csv').exists()

def test_valid_patch_npz_roundtrip(tmp_path):
    p=patches.save_patch_set(arrays(),tmp_path/'patches.npz')
    saved=patches.load_patch_set(p)
    for k,v in arrays().items():np.testing.assert_array_equal(saved[k],v)

@pytest.fixture
def cache(tmp_path,monkeypatch):
    from soccertrack import config
    monkeypatch.setattr(config,'config_sha256',lambda:'d'*64)
    monkeypatch.setattr(safety,'ROOT',tmp_path)
    run=tmp_path/'source';run.mkdir()
    cfg=E.load_evaluation_config();cfg['yolo']=yaml.safe_load((ROOT/'configs/yolo.yaml').read_text())
    folder=run/'required_validation';folder.mkdir()
    pd.DataFrame([[0,0.,0.,2.,2.,.08001],[2,3.,0.,2.,2.,.9]],columns=C.DETECTION_COLUMNS).to_csv(folder/'detections.csv',index=False)
    hashes={'required_validation/detections.csv':digest(folder/'detections.csv')}
    frameids={'eval10':[0],'win60s':[0,1,2]}
    for name,ids in frameids.items():
        p=folder/name/'coverage.json';p.parent.mkdir();p.write_text(json.dumps(dict(frame_ids=ids,decoded_frames=len(ids))))
        hashes[p.relative_to(run).as_posix()]=digest(p)
    source={}
    for rel in ('src/soccertrack/detector_yolo.py','src/soccertrack/tracker_bytetrack.py'):
        dest=run/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes());source[rel]=digest(dest)
    m=dict(split='val',match='117093',weights_sha256='a'*64,selected_configuration=cfg['yolo'],
           dataset_revision=cfg['project']['dataset']['revision'],clip_sha256='b'*64,mot_sha256='c'*64,
           source_sha256=source,required_union_frame_ids=[0,1,2],decoded_union_frames=3,development_frame_ids=[4],
           video_info=dict(width=640,height=360,fps=25.,frame_count_header=3),git_commit='synthetic')
    (run/'manifest.json').write_text(json.dumps(m));(run/'artifact_hashes.json').write_text(json.dumps(hashes))
    return run,cfg,tmp_path/'outputs/migrated'

def test_cache_migration_keeps_low_scores_and_replay_uses_readback(cache,monkeypatch):
    run,cfg,target=cache
    before=digest(run/'required_validation/detections.csv')
    paths=migration.migrate(run,target,cfg,'d'*64,'a'*64)
    win=C.read_table(paths['win60s'],'detections')
    assert win.score.tolist()==[.08,.9] and C.read_run_meta(paths['win60s'])['processed_frame_count']==3
    assert digest(run/'required_validation/detections.csv')==before
    from soccertrack import tracker_bytetrack
    def associate(table,frames,settings):
        pd.testing.assert_frame_equal(table,win)
        assert list(frames)==[0,1,2]  # empty frame1 included
        return pd.DataFrame([[2,1,3.,0.,2.,2.,.9]],columns=C.TRACK_COLUMNS)
    monkeypatch.setattr(tracker_bytetrack,'track_sequence',associate)
    p=migration.replay(paths['win60s'],target/'tracks/t.csv',cfg)
    meta=C.read_run_meta(p)
    assert meta['source_detection_sha256']==digest(paths['win60s'])
    assert 'association only' in meta['runtime_scope']
    with pytest.raises(FileExistsError): migration.migrate(run,target,cfg,'d'*64,'a'*64)
    with pytest.raises(ValueError,match='stride'): migration.replay(paths['eval10'],target/'tracks/sparse.csv',cfg)

def test_cache_migration_rejects_before_writes(cache):
    run,cfg,target=cache
    with pytest.raises(ValueError,match='hash'):migration.migrate(run,target,cfg,'0'*64,'a'*64)
    with pytest.raises(ValueError,match='identity'):migration.migrate(run,target,cfg,'d'*64,'e'*64)
    altered=deepcopy(cfg);altered['yolo']['tiling']['overlap_px']=64
    with pytest.raises(ValueError,match='Configuration'):migration.migrate(run,target,altered,'d'*64,'a'*64)
    p=run/'required_validation/win60s/coverage.json';p.write_text('{}')
    with pytest.raises(ValueError,match='Coverage hash'):migration.migrate(run,target,cfg,'d'*64,'a'*64)
    assert not target.exists()

def test_derivative_guards_precede_inference_or_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(safety,'ROOT',tmp_path)
    with pytest.raises(ValueError): safety.output_directory(tmp_path/'outside')
    with pytest.raises(ValueError): safety.output_directory(tmp_path/'outputs/../../outside')
    run=tmp_path/'outputs/old';run.mkdir(parents=True)
    for name in ('figures','warmed_runtime_probe.csv','member2_yolo_pipeline_experiments_executed.ipynb'):
        (run/name).write_text('preserved')
    from demos.member2.build_artifacts import build
    from demos.member2.runtime_probe import probe
    from demos.member2.build_notebook import build as notebook
    for fn in (build,probe,notebook):
        with pytest.raises(FileExistsError): fn(run)
    assert all(p.read_text()=='preserved' for p in run.iterdir())

def test_notebook_recipe_dependency_order():
    source=(ROOT/'demos/member2/build_notebook.py').read_text(encoding='utf-8-sig')
    start=source.index('To reproduce explicitly')
    recipe=source[start:]
    assert recipe.index('scientific_workflow.py --output') < recipe.index('runtime_probe.py') < recipe.index('build_artifacts.py')

def test_comparison_lineage_and_incompatible_plot(fixtures_dir,tmp_path):
    from soccertrack import compare
    cfg=E.load_evaluation_config();gt=C.read_table(fixtures_dir/'tracks_gt.csv','tracks')
    a=E.evaluate_detections(fixtures_dir/'detections_sample.csv',gt,cfg)
    E.write_metrics(a,tmp_path/'metrics')
    table=compare.detection_table(compare.collect_metrics(tmp_path/'metrics'),'val')
    assert {'protocol','config_sha256','runtime_scope','result_scope'}<=set(table.columns)
    b=deepcopy(a);b['frameset']='other'
    with pytest.raises(ValueError,match='incompatible'):compare.plot_curves([a,b],'pr',tmp_path/'bad.png')
    assert not (tmp_path/'bad.png').exists()
