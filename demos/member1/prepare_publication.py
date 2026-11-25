"""Prepare immutable references for integration; never train or publish official metrics."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from soccertrack import classifiers, features, patches

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {'hog_svm': 'adb3d647191fc770eaa9a65c05717393b4604a4696b49351e4b1f41cb4844304',
            'cnn': '2d25f83d469894a26129aa3e9de7924971fd04a4f8597e170cbe8fd839317643'}

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def prepare(run, destination):
    """Read the trusted expanded run, check checkpoint parity, write a new reference manifest."""
    run=Path(run).resolve(); destination=Path(destination).resolve()
    if not run.is_relative_to(ROOT/'outputs/member1_scientific/runs'):
        raise ValueError('Only trusted local scientific runs may supply pickle checkpoints')
    if not destination.is_relative_to(ROOT/'outputs') or destination == ROOT/'outputs' or destination.exists():
        raise ValueError('A fresh directory inside repository outputs is required')
    m=json.loads((run/'manifest.json').read_text())
    if m['validation_matches'] != ['117093'] or set(m['training_matches']) != {'118575','118576','118577','118578','128058'}:
        raise ValueError('Unexpected source split')
    arrays=patches.load_patch_set(run/'patches_val.npz')
    models={}
    for name,cls,filename in [('hog_svm',classifiers.HogSvmClassifier,'hog_svm.joblib'),('cnn',classifiers.CnnClassifier,'cnn.pt')]:
        path=run/'models'/filename
        actual=digest(path)
        if actual != EXPECTED[name] or actual != m['models'][name]['sha256']:
            raise ValueError('Checkpoint hash mismatch')
        model=cls.load(path)
        scores=model.predict_proba(arrays['X'][:6])
        import pandas as pd
        saved=pd.read_csv(run/f'validation_patch_scores_{name}.csv').score.to_numpy()[:6]
        np.testing.assert_allclose(scores,saved,atol=1e-5,rtol=0)
        if name=='hog_svm':
            assert features.hog_features(arrays['X'][:1],model.cfg).shape==(1,756)
        models[name]={'source':path.relative_to(ROOT).as_posix(),'sha256':actual,'configuration':model.cfg,
                      'six_score_max_delta':float(np.max(np.abs(scores-saved))),
                      'canonical_target':f'outputs/models/scratch_{name}/'+('model.joblib' if name=='hog_svm' else 'model.pt')}
    references={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*')
                if p.is_file() and p.suffix in ('.json','.csv','.npz','.joblib','.pt')}
    document={'status':'PREPARED REFERENCES; Member4 config/path integration and official experiments pending',
              'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
              'source_run':run.relative_to(ROOT).as_posix(),'source_manifest_sha256':digest(run/'manifest.json'),
              'models':models,'candidate_configuration':m['candidate_configuration'],
              'recommended_configuration':m['recommended_configuration'], 'source_artifacts':references,
              'publication_plan':{'eda':'outputs/eda/stats.json and box_sample.csv: export using linear quantiles',
                  'scores':'outputs/scores/<detector>/<split>/patches.csv plus nullable aggregate sidecar',
                  'proposals':'all pre-NMS scored proposals on complete eval10; historical detections insufficient',
                  'detections':'canonical shared writer/readback, exact eval10 and win60s',
                  'tracks':'replay from canonical serialized detections, win60s only'},
              'limitations':['HOG reference parity unresolved; numerical HOG changes require new SVM',
                  'Repeated proposal negatives: nominal3432/704 versus unique2782/580',
                  'Offline noncausal background; validation selection; no held-out result'],
              'retrained':False,'copied_models':False}
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'publication_references.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return document

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('destination');a=p.parse_args()
    prepare(a.run,a.destination)
