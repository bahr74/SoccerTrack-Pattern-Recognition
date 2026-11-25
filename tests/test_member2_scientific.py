"""Scientific diagnostics checked against the independently hand-computed shared fixture."""
from pathlib import Path
import pandas as pd
import pytest
from demos.member2.scientific_workflow import metrics,EXPECTED,CLIP,MOT

ROOT=Path(__file__).resolve().parents[1]

def test_local_counts_against_hand_computed_fixture():
    dets=pd.read_csv(ROOT/'tests/fixtures/detections_sample.csv')
    gt=pd.read_csv(ROOT/'tests/fixtures/tracks_gt.csv')
    result,labels=metrics(dets,gt,[0,1,2],.5)
    assert (result['tp'],result['fp'],result['fn'])==(7,1,2)
    assert result['precision']==pytest.approx(.875)
    assert result['recall']==pytest.approx(7/9)
    assert result['f1']==pytest.approx(14/17)
    assert result['status'].startswith('PROVISIONAL')
    assert sum(label=='fn' for _,label,_ in labels)==2

def test_empty_predictions_and_empty_ground_truth():
    columns=['frame','x','y','w','h','score']
    empty=pd.DataFrame(columns=columns)
    gt=pd.DataFrame([[0,10,10,20,40,1]],columns=columns)
    result,_=metrics(empty,gt,[0,1],.4)
    assert (result['tp'],result['fp'],result['fn'])==(0,0,1)
    result,_=metrics(gt,empty,[0],.4)
    assert (result['tp'],result['fp'],result['fn'])==(0,1,0)

def test_duplicate_detections_cannot_match_same_gt_twice():
    columns=['frame','x','y','w','h','score']
    gt=pd.DataFrame([[0,10,10,20,40,1]],columns=columns)
    dets=pd.concat([gt,gt],ignore_index=True)
    result,_=metrics(dets,gt,[0],.05)
    assert result['tp']==1 and result['fp']==1

def test_development_report_frames_are_disjoint():
    development=set(range(1601,5762,130))
    report=set(range(1500))|set(range(0,5976,10))
    assert len(development)==33 and not development&report
    assert '117093' in str(CLIP) and '117093' in str(MOT)
    assert len(EXPECTED)==64

@pytest.mark.data
def test_saved_full_validation_coverage_and_counts():
    import json
    run=ROOT/'outputs/member2_scientific/runs/member2_validation_20261125'
    manifest=json.loads((run/'manifest.json').read_text())
    from demos.member2.audit import digest
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    assert all(digest(run/path)==expected for path,expected in hashes.items())
    for path,expected in manifest['source_sha256'].items():
        assert digest(run/'source_snapshot'/path)==expected
    assert manifest['split']=='val' and manifest['match']=='117093'
    assert not set(manifest['development_frame_ids'])&set(manifest['required_union_frame_ids'])
    for name,frames in [('eval10',list(range(0,5976,10))),('win60s',list(range(1500)))]:
        coverage=json.loads((run/f'required_validation/{name}/coverage.json').read_text())
        assert coverage['frame_ids']==frames
        dets=pd.read_csv(run/f'required_validation/{name}/detections.csv')
        assert set(dets.frame)<=set(frames)
        assert ((dets.score>=.05)&(dets.score<=1)).all()
        assert ((dets.x>=0)&(dets.y>=0)&(dets.w>1)&(dets.h>1)&(dets.x+dets.w<=4096)&(dets.y+dets.h<=1080)).all()
    tracks=pd.read_csv(run/'required_validation/tracks_starting.csv')
    assert (tracks.track_id>0).all() and not tracks.duplicated(['frame','track_id']).any()
    assert tracks.frame.min()==0 and tracks.frame.max()==1499
    results=manifest['required_detection_results']
    assert [r['frames'] for r in results]==[598,1500]
    for r in results:
        assert r['gt']==22*r['frames'] and r['tp']+r['fn']==r['gt']
        assert r['tp']+r['fp']==r['pred'] and r['f1']==pytest.approx(2*r['tp']/(2*r['tp']+r['fp']+r['fn']))

@pytest.mark.data
def test_saved_notebook_and_video_execution():
    import json
    import nbformat
    run=ROOT/'outputs/member2_scientific/runs/member2_validation_20261125'
    source=nbformat.read(ROOT/'demos/member2/notebooks/member2_yolo_pipeline_experiments.ipynb',as_version=4)
    assert all(c.get('execution_count') is None and not c.get('outputs',[]) for c in source.cells if c.cell_type=='code')
    saved=nbformat.read(run/'member2_yolo_pipeline_experiments_executed.ipynb',as_version=4)
    code=[c for c in saved.cells if c.cell_type=='code']
    assert len(code)>=16 and all(c.execution_count and c.outputs for c in code)
    assert not any(o.output_type=='error' for c in code for o in c.outputs)
    video=json.loads((run/'video_verification.json').read_text())
    assert video['decoded_frames']==150 and video['aspect_preserved']
