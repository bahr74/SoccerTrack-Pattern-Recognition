"""Member 1 train-only calibrated HOG/SVM and bounded small-CNN training."""
from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import random
import numpy as np
from soccertrack import config, features


def _seed(cfg):
    return int(cfg.get('project',{}).get('seed',cfg.get('seed',42)))


def _labels(X,y):
    features._check(X)
    y=np.asarray(y)
    if y.shape!=(len(X),) or set(np.unique(y))!={0,1}:
        raise ValueError('Training/evaluation labels must contain both binary classes')
    return y.astype(np.int8)


class HogSvmClassifier:
    def __init__(self,cfg):
        self.cfg=deepcopy(cfg); self.estimator=None; self.selected_C=None

    def fit(self,X,y):
        y=_labels(X,y)
        return self._fit_hog(features.hog_features(X,self.cfg),y)

    def _fit_hog(self,H,y):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import StratifiedKFold
        folds=min(3,int(np.bincount(y).min()))
        if folds<2:
            raise ValueError('Calibration requires at least two train examples per class')
        self.selected_C=float(self.cfg['svm']['C_grid'][0])
        svm=LinearSVC(C=self.selected_C,class_weight='balanced',max_iter=self.cfg['svm']['max_iter'],random_state=_seed(self.cfg))
        cv=StratifiedKFold(folds,shuffle=True,random_state=_seed(self.cfg))
        self.estimator=CalibratedClassifierCV(svm,method='sigmoid',cv=cv,ensemble=False)
        self.estimator.fit(H,y)
        return self

    def fit_with_validation(self,X,y,X_val,y_val):
        """Additive M1-owned API: choose C by validation AP; calibration stays on train folds."""
        from sklearn.metrics import average_precision_score
        y=_labels(X,y); _labels(X_val,y_val)
        H=features.hog_features(X,self.cfg); Hv=features.hog_features(X_val,self.cfg)
        best=None; score=-np.inf; search=[]
        for C in self.cfg['svm']['C_grid']:
            candidate=HogSvmClassifier(self.cfg)
            candidate.cfg['svm']['C_grid']=[C]
            candidate._fit_hog(H,y)
            ap=average_precision_score(y_val,candidate.estimator.predict_proba(Hv)[:,1])
            search.append({'C':float(C),'validation_pr_auc':float(ap)})
            if ap>score:
                best=candidate; score=ap
        self.estimator=best.estimator; self.selected_C=best.selected_C
        self.validation_pr_auc=float(score)
        self.search_results_=search
        return self

    def predict_proba(self,X):
        if self.estimator is None:
            raise RuntimeError('SVM has not been fitted')
        features._check(X)
        if not len(X): return np.empty(0,dtype=np.float32)
        return self.estimator.predict_proba(features.hog_features(X,self.cfg))[:,1].astype(np.float32)

    def hog_weights(self):
        if self.estimator is None: raise RuntimeError('SVM has not been fitted')
        return self.estimator.calibrated_classifiers_[0].estimator.coef_[0].astype(np.float32).copy()

    def save(self,path):
        import joblib
        if self.estimator is None: raise RuntimeError('SVM has not been fitted')
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({'cfg':self.cfg,'estimator':self.estimator,'selected_C':self.selected_C,'search_results':getattr(self,'search_results_',[])},path)

    @classmethod
    def load(cls,path):
        import joblib
        state=joblib.load(path)  # trusted, locally generated checkpoints only
        obj=cls(state['cfg']); obj.estimator=state['estimator']; obj.selected_C=state['selected_C']
        obj.search_results_=state.get('search_results',[])
        return obj


class CnnClassifier:
    def __init__(self,cfg):
        import torch
        from torch import nn
        self.cfg=deepcopy(cfg); settings=cfg['cnn']
        seed=_seed(cfg); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        choice=settings['device']
        self.device=torch.device('cuda' if choice=='auto' and torch.cuda.is_available() else ('cpu' if choice=='auto' else choice))
        layers=[]; previous=3
        for channels in settings['channels']:
            layers += [nn.Conv2d(previous,channels,3,padding=1),nn.BatchNorm2d(channels),nn.ReLU(),nn.MaxPool2d(2)]
            previous=channels
        layers += [nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Dropout(settings['dropout']),nn.Linear(previous,1)]
        self.model=nn.Sequential(*layers).to(self.device)
        self.gradcam_layer=self.model[8]
        if sum(p.numel() for p in self.model.parameters())>=100000:
            raise ValueError('Small CNN must contain fewer than 100000 parameters')
        self.mean=np.zeros(3,dtype=np.float32); self.std=np.ones(3,dtype=np.float32)
        self.trained=False; self.history={}

    def preprocess(self,X):
        import torch
        features._check(X)
        data=(X.astype(np.float32)/255-self.mean)/self.std
        return torch.from_numpy(data.transpose(0,3,1,2).copy()).to(self.device)

    def fit(self,X_train,y_train,X_val,y_val):
        import torch
        from sklearn.metrics import average_precision_score
        y_train=_labels(X_train,y_train); _labels(X_val,y_val)
        settings=self.cfg['cnn']; seed=_seed(self.cfg)
        self.mean=(X_train.astype(np.float32)/255).mean(axis=(0,1,2))
        self.std=np.maximum((X_train.astype(np.float32)/255).std(axis=(0,1,2)),1/255)
        optimizer=torch.optim.Adam(self.model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
        criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y_train==0).sum()/max((y_train==1).sum(),1)),device=self.device))
        val=self.preprocess(X_val); targets_val=torch.tensor(y_val,dtype=torch.float32,device=self.device)
        history={key:[] for key in ('epoch','train_loss','val_loss','val_pr_auc')}
        best=(-np.inf,-np.inf); best_state=None; stale=0; rng=np.random.default_rng(seed)
        for epoch in range(int(settings['epochs'])):
            self.model.train(); total=0
            order=rng.permutation(len(X_train))
            for start in range(0,len(order),int(settings['batch_size'])):
                ids=order[start:start+int(settings['batch_size'])]
                batch=X_train[ids].copy(); aug=settings['augment']
                if aug.get('hflip',False):
                    flip=rng.random(len(batch))<.5; batch[flip]=batch[flip,:,::-1]
                brightness=float(aug.get('brightness',0)); contrast=float(aug.get('contrast',0))
                scale=rng.uniform(1-contrast,1+contrast,(len(batch),1,1,1))
                shift=rng.uniform(-brightness,brightness,(len(batch),1,1,1))*255
                batch=np.clip(batch*scale+shift,0,255).astype(np.uint8)
                values=self.preprocess(batch); targets=torch.tensor(y_train[ids],dtype=torch.float32,device=self.device)
                optimizer.zero_grad(); loss=criterion(self.model(values).flatten(),targets)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite CNN training loss')
                loss.backward(); optimizer.step(); total+=float(loss.detach())*len(ids)
            self.model.eval()
            with torch.no_grad():
                logits=self.model(val).flatten(); val_loss=float(criterion(logits,targets_val)); probabilities=logits.sigmoid().cpu().numpy()
            ap=float(average_precision_score(y_val,probabilities))
            for key,value in zip(history,(epoch+1,total/len(X_train),val_loss,ap)):
                history[key].append(value)
            key=(ap,-val_loss)  # validation loss breaks equal-AP ties
            if key>best:
                best=key; best_state=deepcopy(self.model.state_dict()); stale=0
            else:
                stale+=1
            if stale>=int(settings['early_stopping_patience']): break
        if best_state is None: raise ValueError('epochs must be positive')
        self.model.load_state_dict(best_state); self.model.eval(); self.trained=True; self.history=history
        return history

    def predict_proba(self,X):
        import torch
        if not self.trained: raise RuntimeError('CNN is untrained; train or load a checkpoint first')
        features._check(X)
        result=[]; self.model.eval()
        with torch.no_grad():
            for start in range(0,len(X),self.cfg['cnn']['batch_size']):
                result.extend(self.model(self.preprocess(X[start:start+self.cfg['cnn']['batch_size']])).flatten().sigmoid().cpu().tolist())
        return np.asarray(result,dtype=np.float32)

    def save(self,path):
        import torch
        if not self.trained: raise RuntimeError('CNN is untrained')
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'cfg':self.cfg,'weights':self.model.state_dict(),'mean':self.mean.tolist(),
                    'std':self.std.tolist(),'history':self.history},path)

    @classmethod
    def load(cls,path):
        import torch
        state=torch.load(path,map_location='cpu',weights_only=True)
        cfg=deepcopy(state['cfg']); cfg['cnn']['device']='cpu'
        obj=cls(cfg); obj.model.load_state_dict(state['weights'])
        obj.mean=np.asarray(state['mean'],dtype=np.float32); obj.std=np.asarray(state['std'],dtype=np.float32)
        obj.history=state['history']; obj.trained=True; obj.model.eval()
        return obj


def load_classifier(detector_id):
    if detector_id not in ('scratch_hog_svm','scratch_cnn'): raise ValueError('Unknown scratch classifier')
    path=config.repo_root()/'outputs/models'/detector_id/('model.joblib' if detector_id=='scratch_hog_svm' else 'model.pt')
    return (HogSvmClassifier if detector_id=='scratch_hog_svm' else CnnClassifier).load(path)
