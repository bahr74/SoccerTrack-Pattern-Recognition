"""Member 1 corrective regression: cache identity, provenance, transport and real parity."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import Mock
import sys,json,threading,http.client,hashlib
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import numpy as np,pandas as pd,cv2,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demos/member1'))
import app,experiment,scientific_workflow as w
from preview import Preview,CancelledOperation,identity
from provenance import indices,resolved_plan,load_plan,selection_metadata,patch_summary

@pytest.fixture
def preview(tmp_path):
    path=tmp_path/'fixture.avi';gt=experiment.make_synthetic(path,40);p=Preview();p.open(path,background_window=12)
    try:yield p
    finally:p.close()

def test_display_views_skip_inference_and_detection_skips_association(preview,monkeypatch):
    monkeypatch.setattr(preview,'_model',Mock(side_effect=AssertionError('No classifier for Original/GT')))
    original=preview.frame(19,'original');assert original['work']['inference']==0 and original['work']['association']==0
    assert preview.cfg['sampling']['background_num_frames']==9 and preview.cfg['detector']['operating_threshold']==.5
    assert original['provenance']['view']=='original'
    preview.gt=pd.DataFrame([[19,1,5,5,10,20,1]],columns=['frame','track_id','x','y','w','h','score'])
    gt=preview.frame(19,'gt');assert gt['work']['inference']==0 and gt['work']['association']==0
    monkeypatch.setattr(preview,'_model',lambda *a:None)
    det=preview.frame(19,'rules');assert det['work']['inference']==1 and det['work']['association']==0
    again=preview.frame(19,'rules');assert again['cache_hit'] and again['processing_fps'] is None
    preview.close();preview.originals.clear();capture=Mock();capture.grab.return_value=False;preview.capture=capture;preview.decode_index=0
    with pytest.raises(ValueError,match='decoding ended'):preview.frame(19,'original')
    capture.release.assert_called_once();assert preview.capture is None

def test_configuration_background_video_identity_invalidates_caches(preview,monkeypatch):
    first=preview.frame(7,'rules');old_resolve=preview.resolve
    def changed(source,method='rules'):
        s=old_resolve(source,method);s['cfg']['segmentation']['close_ksize']=5;s['config_id']=identity(s['cfg']);return s
    monkeypatch.setattr(preview,'resolve',changed)
    next_frame=preview.frame(7,'rules');assert next_frame['provenance']['pipeline_id']!=first['provenance']['pipeline_id']
    assert next_frame['work']['segmentation']==1 and next_frame['work']['inference']==1
    preview.background_id='changed_background';third=preview.frame(7,'rules');assert third['work']['inference']==1
    assert third['provenance']['pipeline_id']!=next_frame['provenance']['pipeline_id']
    preview.video_id='changed_video';fourth=preview.frame(7,'rules');assert fourth['work']['inference']==1
    def incompatible(source,method='rules'):
        s=old_resolve(source,method);s['cfg']['segmentation']['work_scale']=1.;s['config_id']=identity(s['cfg']);return s
    monkeypatch.setattr(preview,'resolve',incompatible)
    with pytest.raises(ValueError,match='work scale'):preview.frame(7,'rules')

def test_tracking_rewind_cache_and_checkpoint_parity(preview):
    forward=preview.frame(35,'tracks');assert forward['work']['association']==36
    rewind=preview.frame(7,'tracks');assert rewind['work']['association']==0
    # Evict only row cache to force restoration of the saved frame-24 association state.
    preview.track_rows.clear();restored=preview.frame(27,'tracks');assert restored['work']['association']==3
    direct=Preview();direct.open(preview.path,background_window=12,cached_background=preview.bg)
    try:expected=direct.frame(27,'tracks');assert restored['track_rows']==expected['track_rows']
    finally:direct.close()
    assert len(preview.originals)<=2 and len(preview.intermediates)<=2 and len(preview.checkpoints)<=8
    preview.track_rows.clear();exact_checkpoint=preview.frame(24,'tracks')
    assert exact_checkpoint['frame']==24 and exact_checkpoint['state']=='Ready'

def test_cancelled_queued_operation_is_not_reported_ready(preview):
    preview.cancel('cancelled-request')
    with pytest.raises(CancelledOperation):preview.frame(7,'rules',operation_id='cancelled-request')
    assert preview.frame(7,'rules',operation_id='new-request')['state']=='Ready'

def test_static_json_cache_avoids_reparsing(tmp_path):
    path=tmp_path/'settings.json';path.write_text('{"v":1}')
    p=Preview();one=p._json(path);one['v']=8;assert p._json(path)=={'v':1}
    assert len(p.json_cache)==1
    path.write_text('{"v":22}');assert p._json(path)=={'v':22}

def test_upload_preflight_and_oversized_body_receive_useful_error(tmp_path,monkeypatch):
    monkeypatch.setattr(w,'BASE',tmp_path/'isolated');server=app.make_server(0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
        conn.request('POST','/api/upload?name=large.mp4',b'x'*(4*1024*1024),headers={'X-Demo-Token':server.demo_token,'Content-Length':str(app.MAX_UPLOAD+1)})
        response=conn.getresponse();body=json.loads(response.read());conn.close();assert response.status==413 and body['state']=='Error' and '512 MiB' in body['error']
        data=json.dumps({'name':'large.mp4','size':app.MAX_UPLOAD+1}).encode();request=Request(f'http://127.0.0.1:{server.server_port}/api/upload-check',data,headers={'X-Demo-Token':server.demo_token})
        with pytest.raises(HTTPError) as e:urlopen(request)
        assert e.value.code==413 and not (w.BASE/'uploads').exists()
        assert 'Error' in server.operation_log.read_text()
    finally:server.shutdown();server.preview.close();server.server_close();thread.join(5)

def make_plan():
    cfg=experiment.demo_config();cfg['sampling']['background_num_frames']=9
    cfg['project']['split'].update(train=['118575'],val=['117093'])
    frames={'118575':[0,20],'117093':[0,10]};bg={k:{'total_frames':40,'sample_indices':indices(40,9),'effective_count':9} for k in frames}
    return resolved_plan(cfg,frames,bg)

def test_resolved_plan_hash_actual_indices_and_leakage(tmp_path):
    plan=make_plan();path=tmp_path/'settings.json';path.write_text(json.dumps(plan));assert load_plan(path)==plan
    bad=deepcopy(plan);bad['backgrounds']['117093']['effective_count']=60;path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='hash'):load_plan(path)
    cfg=plan['configuration'];frames={'128057':[0]}
    with pytest.raises(PermissionError):resolved_plan(cfg,frames,{'128057':plan['backgrounds']['117093']})

def test_selected_frame_metadata_and_duplicate_policy():
    cfg=experiment.demo_config();records=[{'match':'118575','frame':f} for f in [0,750,1500]]
    meta=selection_metadata(records,cfg);assert meta['sampling_stride'] is None and meta['eligibility_stride']==10
    assert meta['selected_frame_ids']=={'118575':[0,750,1500]} and meta['derived_seeds']['118575']==118617
    arrays={'X':np.zeros((3,64,32,3),np.uint8),'y':np.array([1,0,0]),'kind':np.array(['gt','proposal','proposal'])}
    summary=patch_summary(arrays);assert summary['unique']==1 and summary['repeated_copies']==2 and summary['proposal_unique']==1

def test_prepare_only_replays_saved_settings_without_current_yaml_or_training(tmp_path,monkeypatch):
    plan=make_plan();plan['frame_plan']={k:[0] for k in plan['frame_plan']}
    plan=resolved_plan(plan['configuration'],plan['frame_plan'],plan['backgrounds']);path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    monkeypatch.setattr(w,'BASE',tmp_path/'runs_root');monkeypatch.setattr(w,'demo_config',Mock(side_effect=AssertionError('Must use saved config')));monkeypatch.setattr(w,'load_sources',lambda:{})
    cache=tmp_path/'cache';cache.mkdir();cv2.imwrite(str(cache/'background.png'),np.zeros((10,10,3),np.uint8))
    def source(match,sources,cfg):
        gt=pd.DataFrame({'frame':[0,0],'track_id':[1,2],'x':[10.,20.],'y':[20.,60.],'w':[10.,15.],'h':[20.,30.],'score':[1.,1.]})
        return {'match':match,'info':{'frame_count_header':40},'gt':gt,'quality':{'split':'train' if match=='118575' else 'val','annotation_sha256':'a'*64},'cache':cache}
    def records(data,cfg,size,frames):return ([{'patch':np.zeros((64,32,3),np.uint8),'label':int(i==0),'box':[0.,0.,10.,20.],'kind':'gt' if i==0 else 'random','match':data['match'],'frame':0} for i in range(4)],[])
    monkeypatch.setattr(w,'source_data',source);monkeypatch.setattr(w,'collect_dataset',records);monkeypatch.setattr(w,'train_and_report',Mock(side_effect=AssertionError('No retraining')))
    run=w.run_experiment(replay_settings=path,prepare_only=True);assert not (run/'models').exists()
    assert w.read_json(run/'resolved_settings.json')['configuration']==plan['configuration']

@pytest.fixture(scope='module')
def real_evidence():
    import torch
    torch.set_num_threads(2)
    try:run=w.active_run();m=w.read_json(run/'manifest.json');registry=w.read_json(w.BASE/'local_sources.json')['matches']
    except FileNotFoundError:pytest.skip('Approved local validation evidence unavailable')
    item=next(d for d in m['dataset'] if d['match']=='117093');p=Preview();p.open(registry['117093']['clip'],registry['117093']['mot'],5976,cv2.imread(str(w.ROOT/item['background'])),source='expanded',method='hog_svm')
    try:yield p,run,m
    finally:p.close()

@pytest.mark.data
@pytest.mark.parametrize('index',[7,16,17,18,19])
def test_real_same_frame_predictions_and_track_inputs_match_saved_candidate(real_evidence,index):
    p,run,m=real_evidence;expected=pd.read_csv(run/'detection_candidate/hog_svm_detections_local.csv').query('frame==@index').reset_index(drop=True)
    result=p.frame(index,'hog_svm','expanded','hog_svm',640);actual=pd.DataFrame(result['detection_rows'])
    assert len(actual)==len(expected);np.testing.assert_allclose(actual.to_numpy(),expected.to_numpy(),atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(pd.DataFrame(result['track_input_rows']).to_numpy(),expected.loc[expected.score>=.5].to_numpy(),atol=1e-6,rtol=1e-6)
    track=p.frame(index,'tracks','expanded','hog_svm',640);expected_tracks=pd.read_csv(run/'tracking_local.csv').query('frame==@index').reset_index(drop=True)
    np.testing.assert_allclose(pd.DataFrame(track['track_rows']).to_numpy(),expected_tracks.to_numpy(),atol=1e-6,rtol=1e-6)
    assert np.array_equal(p.pitch,experiment.segmentation.pitch_mask(p.bg,p.cfg,'117093'))

@pytest.mark.data
def test_real_api_export_matches_analyzer_and_saved_candidate(real_evidence,tmp_path,monkeypatch):
    p,run,m=real_evidence;monkeypatch.setattr(app,'OUT',tmp_path/'export_root');server=app.make_server(0);worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'
    def post(endpoint,data):
        request=Request(base+'/api/'+endpoint,json.dumps(data).encode(),headers={'Content-Type':'application/json','X-Demo-Token':server.demo_token})
        with urlopen(request,timeout=90) as response:return json.load(response)
    try:
        post('real',{'source':'expanded','method':'hog_svm'});frame=post('frame',{'index':19,'mode':'tracks','source':'expanded','method':'hog_svm'})
        exported=post('export',{'source':'expanded','method':'hog_svm'});assert exported['provenance']['pipeline_id']==frame['provenance']['pipeline_id']
        actual=pd.read_csv(app.OUT/exported['predictions']);expected=pd.read_csv(run/'detection_candidate/hog_svm_detections_local.csv')
        np.testing.assert_allclose(actual.loc[actual.frame<20].to_numpy(),expected.to_numpy(),atol=1e-6,rtol=1e-6)
        track=pd.read_csv(app.OUT/exported['tracks_file']);expected_track=pd.read_csv(run/'tracking_local.csv')
        np.testing.assert_allclose(track.loc[track.frame<20].to_numpy(),expected_track.to_numpy(),atol=1e-6,rtol=1e-6)
        assert exported['export_frames_decoded']==30
    finally:server.shutdown();server.preview.close();server.server_close();worker.join(5)

def test_percentile_linear_convention_matches_shared_fixture(fixtures_dir):
    from soccertrack import data_io
    fixture=json.loads((fixtures_dir/'eda_stats_sample.json').read_text())
    clean,_=data_io.clean_mot_geometry(data_io.read_mot_raw(fixtures_dir/'mot_raw_dirty.txt'),(640,360),3)
    assert len(clean)==11
    assert fixture['matches']['999002']['box_width_px']['p05']==14.0
    assert data_io._box_quantiles(clean.w)['p05']==14.0
    # rank(11-1)*.05=.5: halfway between widths10 and18. No real result is rewritten.
