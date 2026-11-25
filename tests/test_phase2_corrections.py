"""Independent regression cases from the Phase2 integration audit (Member3)."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from soccertrack import contracts as C, evaluate as E, metrics as M

@pytest.fixture
def cfg():
    return E.load_evaluation_config()

@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_evaluation.py'
    spec = importlib.util.spec_from_file_location('guard_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_no_read_runner_authorization(runner, monkeypatch, cfg):
    def denied(*args, **kwargs):
        pytest.fail('A denied request reached data resolution/dispatch')
    monkeypatch.setattr(runner, 'ground_truth', denied)
    for command in ('detection', 'oracle', 'compare'):
        monkeypatch.setattr(runner, 'cmd_' + command, denied)
        args = [command, '--split', 'test']
        if command == 'detection': args += ['--detector', 'yolov8n_coco']
        with pytest.raises(PermissionError): runner.main(args)
    monkeypatch.setattr(runner.E, 'load_evaluation_config', lambda: cfg)
    with pytest.raises(PermissionError):
        runner.main(['oracle', '--split', 'val', '--match', '128057'])
    with pytest.raises(ValueError, match='disabled'):
        runner.main(['compare', '--match', '117092'])

@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_geometry_and_metadata(fixtures_dir, value):
    det = C.read_table(fixtures_dir/'detections_sample.csv', 'detections')
    with pytest.raises(C.ContractError): C.validate_detections(det.assign(x=value))
    meta = C.read_run_meta(fixtures_dir/'detections_sample.csv')
    for key in ('fps', 'runtime_seconds', 'frames_per_second'):
        with pytest.raises(C.ContractError): C.validate_run_meta(dict(meta, **{key: value}))

@pytest.mark.parametrize('value', [1.00001, -0.00001, np.inf, np.nan])
def test_writer_validates_raw_score_before_rounding(fixtures_dir, tmp_path, value):
    det = C.read_table(fixtures_dir/'detections_sample.csv', 'detections').assign(score=value)
    path = tmp_path/'bad.csv'
    with pytest.raises(C.ContractError): C.write_table(det, path, 'detections')
    assert not path.exists()

@pytest.mark.parametrize('number', [2**53+1, 2**63-1])
def test_exact_integer_roundtrip(fixtures_dir, tmp_path, number):
    t = C.read_table(fixtures_dir/'tracks_gt.csv', 'tracks').iloc[:1].assign(track_id=number)
    p = C.write_table(t, tmp_path/'track.csv', 'tracks')
    assert C.read_table(p, 'tracks').track_id.iloc[0] == number

@pytest.mark.parametrize('token', ['9007199254740993.1', str(2**63), 'NaN', 'Infinity', 'true'])
def test_invalid_integer_csv(tmp_path, token):
    path = tmp_path/'track.csv'
    path.write_text('frame,track_id,x,y,w,h,score\n0,'+token+',0,0,2,2,1\n')
    with pytest.raises(C.ContractError): C.read_table(path, 'tracks')

def test_empty_and_malformed_tables_metadata(tmp_path, fixtures_dir):
    for kind in ('detections', 'tracks', 'proposals', 'patch_scores'):
        p = C.write_table(pd.DataFrame(columns=C.TABLE_SCHEMAS[kind]), tmp_path/(kind+'.csv'), kind)
        assert C.read_table(p, kind).empty
    p = tmp_path/'bad.csv'
    p.write_text('frame,x,y,w,h,score\n0,0,0,2,2,1,extra\n')
    with pytest.raises(C.ContractError): C.read_table(p, 'detections')
    C.meta_path(p).write_text('[]')
    with pytest.raises(C.ContractError): C.read_run_meta(p)
    C.meta_path(p).write_text('{broken')
    with pytest.raises(C.ContractError): C.read_run_meta(p)
    meta = C.read_run_meta(fixtures_dir/'detections_sample.csv')
    with pytest.raises(C.ContractError): C.validate_run_meta(dict(meta, created_utc='2026-02-30T00:00:00Z'))

@pytest.mark.parametrize('change', [
    {'score_floor': .5}, {'frameset': 'eval10'},
    {'frames': {'start': 0, 'stop': 2, 'stride': 1}},
    {'processed_frame_count': 2}, {'processed_frame_ids': [0,2]}, {'kind': 'proposals'},
    {'operating_threshold': None},
])
def test_protocol_rejects_mislabeled_runs(fixtures_dir, cfg, change):
    meta = C.read_run_meta(fixtures_dir/'detections_sample.csv')
    gt = C.read_table(fixtures_dir/'tracks_gt.csv', 'tracks')
    with pytest.raises(C.ContractError): E.validate_protocol(dict(meta, **change), cfg, 'detections', gt)

def test_protocol_checks_split_threshold_and_empty_frames(fixtures_dir, cfg, tmp_path):
    meta = C.read_run_meta(fixtures_dir/'detections_sample.csv')
    gt = C.read_table(fixtures_dir/'tracks_gt.csv', 'tracks')
    wrong = dict(meta, match='128057', frameset='smoke')
    with pytest.raises(PermissionError): E.validate_protocol(wrong, cfg, 'detections', gt)
    selected = deepcopy(cfg); selected['yolo'] = {'operating_threshold': .7}
    with pytest.raises(C.ContractError, match='threshold'): E.validate_protocol(meta, selected, 'detections', gt)
    # Empty frame1 is still processed: all9 GT are counted, zero predictions =9 FN.
    p = C.write_table(pd.DataFrame(columns=C.DETECTION_COLUMNS), tmp_path/'empty.csv', 'detections')
    C.write_run_meta(meta, p)
    doc = E.evaluate_detections(p, gt, cfg)
    assert doc['counts'] == dict(frames=3, gt=9, pred=0, tp=0, fp=0, fn=9)

def test_noise_tiny_border_determinism_and_drops():
    gt = pd.DataFrame([[0,1,98.,98.,2.,2.,1.]], columns=C.TRACK_COLUMNS)
    meta = dict(image_width=100, image_height=100, frames=dict(start=0,stop=1,stride=1),score_floor=.05)
    for seed in range(30):
        d = E.noisy_gt_detections(gt, 0, 2, seed, (100,100))
        C.validate_detections(d, meta)
        pd.testing.assert_frame_equal(d, E.noisy_gt_detections(gt, 0, 2, seed, (100,100)))
    assert E.noisy_gt_detections(gt, 1, 2, 0, (100,100)).empty
    assert E.noisy_gt_detections(gt.iloc[:0], 0, 2, 0, (100,100)).empty
    with pytest.raises(ValueError): E.noisy_gt_detections(gt, 1.1, 2, 0)

def test_independent_iou_ap_classification_and_matching():
    np.testing.assert_allclose(M.iou_matrix(np.array([[0,0,10,10]]), np.array([[5,0,10,10],[20,0,10,10]])), [[1/3,0]])
    ap, _ = M.average_precision(np.array([True,False,True]), np.array([.9,.8,.7]), 2)
    assert ap == pytest.approx(5/6)
    m = M.binary_classification_metrics(np.array([1,1,0,0]), np.array([.9,.2,.8,.1]), .5)
    assert m['counts'] == dict(tp=1,fp=1,tn=1,fn=1)
    assert m['metrics']['f1'] == .5
    r = M.match_frame(np.array([[0,0,10,10]]), np.array([[0,0,10,10],[0,0,10,10]]), np.array([.9,.8]), .5)
    assert r['tp_flags'].tolist() == [True,False]

def test_pool_rejects_incompatible_protocol(fixtures_dir, cfg):
    gt = C.read_table(fixtures_dir/'tracks_gt.csv', 'tracks')
    a = E.evaluate_detections(fixtures_dir/'detections_sample.csv', gt, cfg)
    for key, value in [('iou_threshold', .3), ('score_threshold', .9), ('source_config_sha256', 'a'*64)]:
        b = deepcopy(a); b['match']='999002'; b['run_id']=b['run_id'].replace('999001','999002'); b['params'][key]=value
        with pytest.raises(ValueError, match='protocol'): E.pool_metrics([a,b])
    b=deepcopy(a);b['match']='999002';b['run_id']=b['run_id'].replace('999001','999002')
    b['params']['frames']['stride']=2
    with pytest.raises(ValueError,match='stride'): E.pool_metrics([a,b])

def test_protocol_config_hash_and_iou_rejection(fixtures_dir,cfg,monkeypatch):
    from soccertrack import config
    meta=C.read_run_meta(fixtures_dir/'detections_sample.csv')
    meta.update(match='117093',frameset='smoke',source_num_frames=3)
    monkeypatch.setattr(config,'config_sha256',lambda:'f'*64)
    with pytest.raises(C.ContractError,match='hash'):E.validate_protocol(meta,cfg,'detections')
    bad=deepcopy(cfg);bad['evaluation']['detection']['iou_thresholds']=[.3,.5]
    meta['config_sha256']='f'*64
    with pytest.raises(C.ContractError,match='IoU'):E.validate_protocol(meta,bad,'detections')

def test_noisy_false_positive_and_proposal_coverage(fixtures_dir,cfg):
    gt=pd.DataFrame([[0,1,20.,20.,2.,2.,1.]],columns=C.TRACK_COLUMNS)
    det=E.noisy_gt_detections(gt,0,10,0,(100,100))
    # Seed0's y jitter moves the tiny box by >6 pixels, so its height2 has no overlap.
    r=M.match_frame(gt[['x','y','w','h']].to_numpy(),det[['x','y','w','h']].to_numpy(),det.score.to_numpy(),.5)
    assert r['tp_flags'].tolist()==[False] and not r['gt_matched'].any()
    fullgt=C.read_table(fixtures_dir/'tracks_gt.csv','tracks')
    doc=E.evaluate_proposals(fixtures_dir/'proposals_sample.csv',fullgt,cfg)
    assert doc['breakdown']['proposal_recall_at_iou']['0.5']==round(4/9,4)

def test_declared_source_count_cannot_shorten_full_gt(fixtures_dir,cfg):
    meta=C.read_run_meta(fixtures_dir/'detections_sample.csv')
    meta.update(source_num_frames=2)
    gt=C.read_table(fixtures_dir/'tracks_gt.csv','tracks')
    with pytest.raises(C.ContractError,match='truncates'): E.validate_protocol(meta,cfg,'detections',gt)

def test_pooled_ap_rejects_mixed_thresholds(fixtures_dir,cfg,tmp_path):
    gt=C.read_table(fixtures_dir/'tracks_gt.csv','tracks')
    a=fixtures_dir/'detections_sample.csv'
    b=C.write_table(C.read_table(a,'detections'),tmp_path/'b.csv','detections')
    C.write_run_meta(dict(C.read_run_meta(a),operating_threshold=.9),b)
    with pytest.raises(ValueError,match='incompatible'): E.pooled_detection_ap([(a,gt),(b,gt)],cfg)

def test_hota_sequence_combination_counterexample(fixtures_dir, cfg, tmp_path):
    local = deepcopy(cfg); local['evaluation']['framesets']['fixture3']['stop']=1
    docs=[]
    for count, predicted, match in [(100, True, '999001'), (1, False, '999002')]:
        gt = pd.DataFrame([[0,i+1,3.*i,0.,2.,2.,1.] for i in range(count)], columns=C.TRACK_COLUMNS)
        p = C.write_table(gt if predicted else gt.iloc[:0], tmp_path/(match+'.csv'), 'tracks')
        meta = C.read_run_meta(fixtures_dir/'tracks_pred.csv')
        meta.update(match=match, frames=dict(start=0,stop=1,stride=1))
        C.write_run_meta(meta,p)
        docs.append(E.evaluate_tracking(p,gt,local,workspace_root=tmp_path/'te'))
    pooled = E.pool_metrics(docs)
    assert pooled['metrics']['hota'] == round(np.sqrt(100/101),4)
    assert pooled['metrics']['mota'] == round(100/101,4)
    assert pooled['metrics']['idf1'] == round(200/201,4)
    assert pooled['metrics']['idsw'] == 0
    assert pooled['metrics']['hota'] != .5
    legacy=deepcopy(docs); del legacy[0]['inputs']['trackeval_statistics']
    with pytest.raises(ValueError, match='statistics'): E.pool_metrics(legacy)
