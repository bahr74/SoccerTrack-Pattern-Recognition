"""Write explicit corrective supplements alongside, never into, a historical run."""
from pathlib import Path
from copy import deepcopy
import json,hashlib
import cv2,numpy as np,pandas as pd
import scientific_workflow as w
from provenance import resolved_plan,indices,patch_summary
from soccertrack import data_io,patches,proposals,segmentation,detector_scratch


def active_folder():return w.ROOT/w.read_json(w.BASE/'corrections/active_fix.json')['path']

def build_supplement(run=None,target=None):
    run=Path(run or w.active_run());target=Path(target or active_folder());target.mkdir(parents=True,exist_ok=True)
    m=w.read_json(run/'manifest.json');cfg=w.read_json(run/'configuration.json');cfg['sampling']['background_num_frames']=9
    backgrounds={d['match']:{'total_frames':d['video_header']['frame_count_header'],'sample_indices':indices(d['video_header']['frame_count_header'],9),'effective_count':9} for d in m['dataset']}
    inputs={d['match']:{'annotation_sha256':d['annotation_sha256'],'background_sha256':w.sha(w.ROOT/d['background'])} for d in m['dataset']}
    plan=resolved_plan(cfg,m['frame_plan'],backgrounds,inputs);w.write_json(target/'historical_resolved_settings.json',plan)
    arrays={name:w.load_patch_arrays(run,name) for name in ('train','val')};summaries={name:patch_summary(a) for name,a in arrays.items()}
    sensitivity={};val=arrays['val'];hashes=np.array([hashlib.sha256(x.tobytes()).hexdigest() for x in val['X']]);unique=np.unique(hashes,return_index=True)[1]
    for name in ['majority_baseline','hog_svm','cnn']:
        score=pd.read_csv(run/f'validation_patch_scores_{name}.csv').score.to_numpy();sensitivity[name]=w.patch_metrics(val['y'][unique],score[unique])
    mcfg=deepcopy(cfg)
    if m['recommended_configuration']=='candidate':
        for key in ['segmentation','proposals','patches','hog','detector','tracker_iou']:mcfg[key]=deepcopy(m['candidate_configuration'][key])
    registry=w.read_json(w.BASE/'local_sources.json')['matches'];clip=w.safe_video(registry['117093']['clip']);image=next(data_io.iter_frames(clip,[0]))[1]
    item=next(d for d in m['dataset'] if d['match']=='117093');bg=cv2.imread(str(w.ROOT/item['background']));pitch=segmentation.pitch_mask(bg,mcfg,'117093')
    p,_,_=detector_scratch.prepare_frame_proposals(image,0,bg,pitch,proposals.SizeModel(**m['size_model']),mcfg)
    # Preserve native RGB crop aspect ratio in a new supplemental contact sheet.
    subset=[patches.crop_patch(image,tuple(box),mcfg['patches']['context']) for box in p[['x','y','w','h']].to_numpy()[:8]]
    canvas=np.zeros((64,32*max(1,len(subset)),3),np.uint8)
    for i,patch in enumerate(subset):canvas[:,32*i:32*(i+1)]=cv2.cvtColor(patch,cv2.COLOR_RGB2BGR)
    stage=target/'presentation/stages';stage.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(stage/'07_patches.png'),canvas)
    supplement={'status':'RETROSPECTIVE AUDIT CLARIFICATION; historical files and measured results are unchanged; no model training',
        'source_run':w.rel(run),'source_manifest_sha256':w.sha(run/'manifest.json'),'resolved_settings':w.rel(target/'historical_resolved_settings.json'),
        'metadata_discrepancies':{'historical_sampling_stride':10,'meaning':'eligibility stride only; actual selected frames are explicit in the plan and NPZ arrays',
            'historical_config_background_count':60,'actual_background_count':9,'master_seed':42,'derived_sampling_seed_rule':'master_seed + integer match ID'},
        'duplication':summaries,'unique_patch_sensitivity_metrics':sensitivity,
        'sensitivity_status':'Sensitivity analysis of existing scores only; altered weighting/class balance; not replacement official metrics',
        'sampling_policy':'Proposal negatives are drawn with replacement; repeated copies retain sample weight. This is not demonstrated train/validation leakage.',
        'future_experiment':'After protocol approval, create a separately versioned experiment comparing replacement, unique sampling and/or explicit weighting on broader validation coverage; do not overwrite historical arrays or weights.',
        'presentation_override':{'stages/07_patches.png':w.rel(stage/'07_patches.png')}}
    w.write_json(target/'historical_audit_supplement.json',supplement);w.write_json(w.BASE/'corrections/latest.json',{'path':w.rel(target),'source_run':w.rel(run)})
    print(target,flush=True);return target

if __name__=='__main__':build_supplement()
