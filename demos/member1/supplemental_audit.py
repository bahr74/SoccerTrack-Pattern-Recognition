"""Paired historical-model and local identity audit on the expanded run's frames."""
from pathlib import Path
import numpy as np,pandas as pd
import scientific_workflow as w
from experiment import OUT,demo_config
from soccertrack import classifiers,proposals,tracker_iou

def audit():
    run=w.active_run();m=w.read_json(run/'manifest.json');cfg=w.read_json(run/'configuration.json')
    selected=m['candidate_configuration'] if m['recommended_configuration']=='candidate' else cfg
    data=w.source_data('117093',w.load_sources(),cfg);size=proposals.SizeModel(**m['size_model'])
    old={'hog_svm':classifiers.HogSvmClassifier.load(OUT/'real_subset_models/hog_svm.joblib'),'cnn':classifiers.CnnClassifier.load(OUT/'real_subset_models/cnn.pt')}
    import torch
    torch.set_num_threads(2)
    summary,_,_,_=w.evaluate_frames(data,selected,size,old,m['report_frames'],run/'earlier_models_same_conditions')
    current=m['paired_detection'][m['recommended_configuration']]
    tracks=pd.read_csv(run/'tracking_local.csv'); gt=data['gt']; assignments=[]
    for frame,group in tracks.groupby('frame'):
        truth=gt.loc[gt.frame==frame]; matched,fp,fn=w.match_frame(group,truth)
        for pred,actual in matched:assignments.append({'frame':int(frame),'predicted_id':int(group.loc[pred,'track_id']),'gt_id':int(truth.loc[actual,'track_id'])})
    rows=pd.DataFrame(assignments,columns=['frame','predicted_id','gt_id']); transitions=0; gaps=0
    for _,group in rows.groupby('gt_id'):
        group=group.sort_values('frame');transitions+=int((group.predicted_id.diff().dropna()!=0).sum());gaps+=int((group.frame.diff().dropna()>1).sum())
    identity={'status':'PROVISIONAL per-frame greedy IoU>=0.5 identity diagnostic; NOT official IDSW/HOTA/IDF1',
        'frames':m['report_frames'],'observed_gt_ids_matched':int(rows.gt_id.nunique()),'gt_ids_in_report':int(gt.loc[gt.frame.isin(m['report_frames'])].track_id.nunique()),
        'matched_track_observations':len(rows),'predicted_id_changes_for_same_gt':transitions,'gaps_between_matched_gt_observations':gaps,
        'interpretation':'Short easy/correlated window can have zero observed changes while missing players or fragmenting outside the window. No appearance or motion model; broader occlusion/crossing audit remains needed.'}
    rows.to_csv(run/'identity_assignments_local.csv',index=False)
    value={'historical_checkpoint_hashes':{name:w.sha(OUT/'real_subset_models'/filename) for name,filename in [('hog_svm','hog_svm.joblib'),('cnn','cnn.pt')]},
        'conditions':'Same report frames 0..19, current train-fitted size model, nine-sample full-window background, selected candidate settings, score 0.5, IoU 0.5. Only classifier checkpoints differ. Earlier historical published number is not a paired comparison.',
        'earlier_pilot_models':summary,'expanded_models':current,'identity_diagnostic':identity}
    w.write_json(run/'supplemental_comparison.json',value);print(identity,flush=True)
    print('Saved',run/'supplemental_comparison.json',flush=True)
    return value
if __name__=='__main__':audit()
