"""Resolved local experiment plans; no shared-schema or historical-file changes."""
from copy import deepcopy
from pathlib import Path
import hashlib,json
import numpy as np
from soccertrack import split

SCHEMA='member1_local_resolved_v1'

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def indices(total,count):
    if not isinstance(total,int) or not isinstance(count,int) or total<1 or count<1:raise ValueError('Background count and total must be positive integers')
    return np.unique(np.linspace(0,total-1,min(count,total),dtype=np.int64)).tolist()

def selection_metadata(records,cfg,seed_rule='master_plus_match_id'):
    actual={}
    for row in records:actual.setdefault(str(row['match']),set()).add(int(row['frame']))
    return {'sampling_stride':None,'eligibility_stride':int(cfg['sampling']['train_frame_stride']),
            'selected_frame_ids':{k:sorted(v) for k,v in sorted(actual.items())},'seed_rule':seed_rule,
            'derived_seeds':{k:int(cfg['project']['seed'])+(int(k) if seed_rule=='master_plus_match_id' else 0) for k in actual},
            'proposal_negative_sampling':'with_replacement; repeated patches retain their sample weights'}

def resolved_plan(cfg,frame_plan,backgrounds,inputs=None):
    value={'schema':SCHEMA,'configuration':deepcopy(cfg),'frame_plan':deepcopy(frame_plan),'backgrounds':deepcopy(backgrounds),
           'inputs':deepcopy(inputs or {}),'master_seed':int(cfg['project']['seed']),
           'derived_sampling_seeds':{k:int(cfg['project']['seed'])+int(k) for k in frame_plan},
           'eligibility_stride':int(cfg['sampling']['train_frame_stride']),'proposal_negative_sampling':'with_replacement'}
    value['settings_sha256']=digest(value);validate_plan(value);return value

def validate_plan(value):
    if value.get('schema')!=SCHEMA:raise ValueError('Unsupported resolved-plan schema')
    expected=value.get('settings_sha256');body={k:v for k,v in value.items() if k!='settings_sha256'}
    if expected!=digest(body):raise ValueError('Resolved settings hash mismatch')
    cfg=value['configuration'];allowed=set(split.matches_for('train',cfg['project'])+split.matches_for('val',cfg['project']))
    if not set(value['frame_plan']).issubset(allowed):raise PermissionError('Replay plan includes a held-out, disabled or unknown match')
    if value['master_seed']!=cfg['project']['seed']:raise ValueError('Master-seed mismatch')
    for match,frames in value['frame_plan'].items():
        bg=value['backgrounds'][match];total=bg['total_frames']
        if any(isinstance(f,bool) or not isinstance(f,int) or not 0<=f<total for f in frames):raise ValueError('Invalid selected frame index')
        if frames!=sorted(set(frames)):raise ValueError('Frame plan must be sorted and unique')
        if any(f%value['eligibility_stride'] for f in frames):raise ValueError('Selected patch frame violates the eligibility stride')
        count=cfg['sampling']['background_num_frames']
        if bg['sample_indices']!=indices(total,count) or bg['effective_count']!=len(bg['sample_indices']):raise ValueError('Background sample metadata mismatch')
        if value['derived_sampling_seeds'][match]!=value['master_seed']+int(match):raise ValueError('Derived sampling-seed mismatch')
    return value

def load_plan(path):return validate_plan(json.loads(Path(path).read_text(encoding='utf-8-sig')))

def patch_summary(arrays):
    hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in arrays['X']];total=len(hashes);unique=len(set(hashes));proposal=np.flatnonzero(arrays['kind']=='proposal')
    return {'total':total,'unique':unique,'repeated_copies':total-unique,'positive':int(arrays['y'].sum()),'negative':int((arrays['y']==0).sum()),
            'proposal_total':len(proposal),'proposal_unique':len({hashes[i] for i in proposal}),
            'interpretation':'Exact within-split duplicates are sampling weights, not demonstrated train/validation leakage. Unique-patch metrics are sensitivity analyses, not replacements for official results.'}
