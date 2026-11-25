"""Bounded, configuration-identified Member 1 analyzer and association replay."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import base64, hashlib, json, threading, time
import cv2
import numpy as np
import pandas as pd
from experiment import ROOT, OUT, demo_config, safe_video, raw_gt, draw
from soccertrack import data_io,segmentation,proposals,detector_scratch,tracker_iou,classifiers,contracts


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def remember(cache,key,value,limit):
    cache[key]=value; cache.move_to_end(key)
    while len(cache)>limit: cache.popitem(last=False)
    return value


class Preview:
    """No timestamp seeks. Image caches are bounded; tracker snapshots contain no images."""
    def __init__(self):
        self.lock=threading.RLock(); self.path=None; self.capture=None; self.iterator=None
        self.models=OrderedDict(); self.json_cache=OrderedDict(); self.file_hashes=OrderedDict()
        self.originals=OrderedDict(); self.masks=OrderedDict(); self.intermediates=OrderedDict(); self.detections=OrderedDict()
        self.checkpoints=OrderedDict(); self.track_rows=OrderedDict(); self.base_signature=None
        self.track_key=None; self.track_index=-1; self.cached=None
        self.cancel_lock=threading.Lock(); self.cancelled=OrderedDict(); self.operation=None

    def begin_operation(self,operation=None):
        with self.cancel_lock:self.operation=operation or str(time.time_ns())
        self.check_cancelled()

    def cancel(self,operation):
        with self.cancel_lock:remember(self.cancelled,operation,True,32)

    def check_cancelled(self):
        with self.cancel_lock:
            if self.operation in self.cancelled:
                self.close()
                raise CancelledOperation('Operation cancelled between processing frames')

    def close(self):
        if self.capture is not None:self.capture.release(); self.capture=None
        self.iterator=None; self.decode_index=-1

    def _json(self,path):
        path=Path(path); stat=path.stat(); key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
        if key not in self.json_cache:remember(self.json_cache,key,json.loads(path.read_text(encoding='utf-8-sig')),16)
        return deepcopy(self.json_cache[key])

    def _hash(self,path):
        path=Path(path); stat=path.stat(); key=(str(path.resolve()),stat.st_mtime_ns,stat.st_size)
        if key not in self.file_hashes:remember(self.file_hashes,key,hashlib.sha256(path.read_bytes()).hexdigest(),16)
        return self.file_hashes[key]

    def _base(self):
        signature=tuple((ROOT/'configs'/name).stat().st_mtime_ns for name in ['project.yaml','scratch.yaml'])
        if signature!=self.base_signature:self.base=demo_config();self.base_signature=signature
        return deepcopy(self.base)

    def resolve(self,source,method='rules'):
        if source not in ('synthetic','real','expanded'):raise ValueError('Unknown model source')
        if method not in ('rules','hog_svm','cnn'):raise ValueError('Unknown detector')
        cfg=self._base(); label='starting / '+source; checkpoint=None; checkpoint_hash=None
        if source=='expanded':
            import scientific_workflow as w
            run=ROOT/self._json(w.BASE/'latest.json')['path']; manifest=self._json(run/'manifest.json')
            cfg=self._json(run/'configuration.json'); label=manifest['recommended_configuration']+' / '+run.name
            selected=manifest.get('candidate_configuration',cfg) if manifest['recommended_configuration']=='candidate' else cfg
            for name in ['segmentation','proposals','patches','hog','detector','tracker_iou']:cfg[name]=deepcopy(selected[name])
            size=proposals.SizeModel(**self._json(run/'size_model.json')); folder=run/'models'
            expected=manifest.get('models',{}).get(method,{}).get('sha256')
        else:
            folder=OUT/('real_subset_models' if source=='real' else 'synthetic_models')
            if (folder/'config.json').is_file():cfg=self._json(folder/'config.json')
            manifest=self._json(OUT/'experiment_provisional.json') if (OUT/'experiment_provisional.json').is_file() else {}
            item=manifest.get('real' if source=='real' else 'synthetic',{})
            size=proposals.SizeModel(**item['size_model']) if 'size_model' in item else proposals.SizeModel(0,45,20/45)
            expected=None
        if method!='rules':
            checkpoint=folder/('hog_svm.joblib' if method=='hog_svm' else 'cnn.pt')
            if not checkpoint.is_file():raise FileNotFoundError(f'Checkpoint unavailable: {checkpoint.relative_to(ROOT)}')
            checkpoint_hash=self._hash(checkpoint)
            if expected and expected!=checkpoint_hash:raise ValueError('Checkpoint SHA-256 differs from the selected run manifest')
        # Resolve inference choices explicitly; the historical training file is unchanged.
        cfg['sampling']['background_num_frames']=9
        cfg['detector']['classifier']=method;cfg['detector']['operating_threshold']=.5
        cfg['tracker_iou']['min_score']=.5;cfg['cnn']['device']='cpu'
        return {'cfg':cfg,'config_id':identity(cfg),'configuration':label,'size':size,'source':source,'method':method,
                'checkpoint':checkpoint,'checkpoint_sha256':checkpoint_hash}

    def activate(self,selection):
        self.selection=selection; self.cfg=selection['cfg']
        if self.path is None:raise ValueError('Select and open a video first')
        scale=self.cfg['segmentation']['work_scale']
        if self.bg.shape!=(int(self.info['height']*scale),int(self.info['width']*scale),3):
            raise ValueError('Selected work scale is incompatible with the loaded background; reopen the video under this preset')
        self.preprocessing_key=identity({'config':selection['config_id'],'background':self.background_id,'video':self.video_id})
        self.pipeline_key=identity({'preprocessing':self.preprocessing_key,'size':selection['size'].__dict__,'method':selection['method'],'model':selection['checkpoint_sha256']})
        self.pitch=self.masks.get(self.preprocessing_key)
        return selection

    def pitch_for(self):
        if self.preprocessing_key not in self.masks:
            remember(self.masks,self.preprocessing_key,segmentation.pitch_mask(self.bg,self.cfg,self.path.stem),4)
        self.pitch=self.masks[self.preprocessing_key];return self.pitch

    def provenance(self):
        s=self.selection
        return {'video':self.path.name,'video_id':self.video_id,'background_id':self.background_id,'configuration':s['configuration'],
                'config_id':s['config_id'],'pipeline_id':self.pipeline_key,'model':s['method'],'source':s['source'],
                'checkpoint':s['checkpoint'].relative_to(ROOT).as_posix() if s['checkpoint'] else None,'checkpoint_sha256':s['checkpoint_sha256'],
                'score_threshold':.5,'threshold_policy':'Fixed preview threshold; not an approved frozen-test operating point',
                'background_sample_indices':self.background_indices,'background_sample_count':len(self.background_indices)}

    def open(self,path,gt_path=None,background_window=60,cached_background=None,source='synthetic',method='rules',operation_id=None):
        self.begin_operation(operation_id)
        self.close();self.path=None;self.cached=None
        for cache in [self.originals,self.masks,self.intermediates,self.detections,self.checkpoints,self.track_rows]:cache.clear()
        self.track_key=None;self.track_index=-1
        selection=self.resolve(source,method) # before background/pitch preprocessing
        candidate=safe_video(path);info=data_io.video_info(candidate)
        if min(info['width'],info['height'])<2 or not np.isfinite(info['fps']) or info['fps']<=0:raise ValueError('Video metadata is invalid or the decoder is unavailable')
        if gt_path and candidate.stem.isdigit() and Path(gt_path).stem!=candidate.stem:raise ValueError('Numeric video and MOT match IDs must agree')
        total=min(info['frame_count_header'],600)
        if total<1:raise ValueError('Video has no usable header frames')
        window=min(info['frame_count_header'],int(background_window))
        if not 1<=window<=6000:raise ValueError('Background window must be within 1..6000 frames')
        gt=raw_gt(gt_path,(info['width'],info['height'])) if gt_path else None
        self.cfg=selection['cfg'];scale=self.cfg['segmentation']['work_scale']
        bg=(cached_background.copy() if cached_background is not None else segmentation.estimate_background(candidate,min(9,window),window,scale))
        expected=(int(info['height']*scale),int(info['width']*scale),3)
        if bg.shape!=expected or bg.dtype!=np.uint8:raise ValueError('Background dimensions do not match the selected configuration and video')
        self.bg=bg;self.background_window=window;self.background_indices=np.unique(np.linspace(0,window-1,min(9,window),dtype=np.int64)).tolist()
        self.background_id=hashlib.sha256(bg.tobytes()+str(bg.shape).encode()).hexdigest()
        stat=candidate.stat();self.video_id=identity({'path':str(candidate),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        self.path=candidate;self.info=info;self.total=total;self.gt=gt;self.activate(selection)
        try:self._original(0,{'decoded':0,'grabs':0,'ids':set()}) # verify real decode, not just headers
        except Exception:self.close();self.path=None;raise
        return dict(info,preview_frames=total,background_window=window,name=candidate.name,ground_truth_available=bool(gt_path),provenance=self.provenance(),label='Historical/live scratch preview; official integrated results use accepted validation provider')

    def _model(self,source,method):
        if method in ('original','gt','pitch','foreground','proposals'):return None
        s=self.resolve(source,method)
        if method=='rules':return None
        key=(str(s['checkpoint']),s['checkpoint_sha256'])
        if key not in self.models:remember(self.models,key,(classifiers.HogSvmClassifier if method=='hog_svm' else classifiers.CnnClassifier).load(s['checkpoint']),4)
        return self.models[key]

    def _size_model(self,source):return self.resolve(source,'rules')['size']

    def _original(self,index,work):
        if index in self.originals:
            self.originals.move_to_end(index);return self.originals[index]
        if self.capture is None or index<=self.decode_index:
            self.close();self.capture=data_io._open_video(self.path);self.decode_index=-1
        while self.decode_index<index:
            self.check_cancelled()
            if not self.capture.grab():
                last=self.decode_index;self.close()
                raise ValueError(f'Video decoding ended before requested frame {index}; last available frame {last}')
            self.decode_index+=1;work['grabs']+=1
        ok,image=self.capture.retrieve()
        if not ok or image is None:
            self.close()
            raise ValueError(f'Failed to decode frame {index}; select a compatible video')
        work['decoded']+=1;work['ids'].add(index)
        remember(self.originals,index,image,2)
        while len(self.originals)>1 and sum(v.nbytes for v in self.originals.values())>64*1024*1024:self.originals.popitem(last=False)
        return image

    def _prepare(self,index,work):
        key=(self.preprocessing_key,identity(self.selection['size'].__dict__),index)
        if key not in self.intermediates:
            image=self._original(index,work);p,raw,mask=detector_scratch.prepare_frame_proposals(image,index,self.bg,self.pitch_for(),self.selection['size'],self.cfg)
            remember(self.intermediates,key,(p,raw,mask),2);work['segmentation']+=1;work['ids'].add(index)
        return self.intermediates[key]

    def _detect(self,index,work):
        key=(self.pipeline_key,index)
        if key not in self.detections:
            p,raw,mask=self._prepare(index,work);image=self._original(index,work)
            scored,d=detector_scratch.classify_candidates(image,p,self.cfg,self._model(self.selection['source'],self.selection['method']))
            remember(self.detections,key,(scored,d),256);work['inference']+=1;work['ids'].add(index)
        return self.detections[key]

    def _track(self,index,work):
        if self.track_key!=self.pipeline_key:
            self.track_key=self.pipeline_key;self.track_index=-1;self.track_rows.clear();self.checkpoints.clear()
            o=self.cfg['tracker_iou'];self.tracker=tracker_iou.IouHungarianTracker(o['iou_gate'],o['n_init'],o['max_age'],.5)
        if index in self.track_rows:return self.track_rows[index]
        if index<self.track_index:
            # Replay the requested frame itself if its row table was evicted.
            available=[i for i in self.checkpoints if i<index]
            if available:
                start=max(available);self.tracker=deepcopy(self.checkpoints[start]);self.track_index=start
            else:
                o=self.cfg['tracker_iou'];self.tracker=tracker_iou.IouHungarianTracker(o['iou_gate'],o['n_init'],o['max_age'],.5);self.track_index=-1
        for f in range(self.track_index+1,index+1):
            self.check_cancelled()
            _,d=self._detect(f,work);rows=self.tracker.update(f,d);remember(self.track_rows,f,rows,256)
            self.track_index=f;work['association']+=1;work['ids'].add(f)
            if f%25==24:remember(self.checkpoints,f,deepcopy(self.tracker),8)
        return self.track_rows[index]

    def frame(self,index,mode='rules',source='synthetic',method='rules',width=1000,operation_id=None):
        self.begin_operation(operation_id)
        if self.path is None:raise ValueError('Select and open a video first')
        if mode not in ('original','gt','pitch','foreground','proposals','rules','hog_svm','cnn','tracks'):raise ValueError('Unknown mode')
        if not 0<=index<self.total or not 320<=width<=1920:raise ValueError('Preview limits exceeded')
        detector=mode if mode in ('rules','hog_svm','cnn') else method
        self.activate(self.resolve(source,detector));start=time.perf_counter()
        work={'decoded':0,'grabs':0,'segmentation':0,'inference':0,'association':0,'ids':set()}
        p=pd.DataFrame(columns=contracts.PROPOSAL_COLUMNS);d=pd.DataFrame(columns=contracts.DETECTION_COLUMNS);tracks=pd.DataFrame(columns=contracts.TRACK_COLUMNS);mask=None
        if mode=='tracks':tracks=self._track(index,work);p,d=self._detect(index,work)
        elif mode in ('rules','hog_svm','cnn'):p,d=self._detect(index,work)
        elif mode in ('foreground','proposals'):p,_,mask=self._prepare(index,work)
        original=self._original(index,work)
        if mode=='original':processed=original
        elif mode=='gt':
            if self.gt is None:raise ValueError('No Ground Truth is available for this uploaded video; use the registered real validation demo')
            processed=draw(original,self.gt.loc[self.gt.frame==index],(0,255,0),'GT')
        elif mode in ('pitch','foreground'):
            small=self.pitch_for() if mode=='pitch' else mask
            processed=cv2.cvtColor(cv2.resize(small.astype(np.uint8)*255,(original.shape[1],original.shape[0]),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
        elif mode=='proposals':processed=draw(original,p,(255,100,0),'Proposal')
        elif mode=='tracks':processed=draw(original,tracks,(0,200,255),'ID')
        else:processed=draw(original,d.loc[d.score>=.5],(0,200,255),detector)
        self.cached=(original,p,d,mask,tracks);self.index=index
        self.check_cancelled()
        elapsed=time.perf_counter()-start;size=(min(width,original.shape[1]),max(1,round(original.shape[0]*min(width,original.shape[1])/original.shape[1])))
        def encoded(image):
            ok,value=cv2.imencode('.jpg',cv2.resize(image,size))
            if not ok:raise RuntimeError('JPEG encoding failed')
            return base64.b64encode(value).decode('ascii')
        processed_count=len(work['ids']);work.pop('ids')
        return {'frame':index,'original':encoded(original),'processed':encoded(processed),'candidates':len(p),'detections':int((d.score>=.5).sum()),'tracks':len(tracks),
                'detection_rows':d.to_dict('records'),'track_input_rows':d.loc[d.score>=.5].to_dict('records'),'track_rows':tracks.to_dict('records'),
                'processing_seconds':elapsed,'processed_frames':processed_count,'cache_hit':processed_count==0,'processing_fps':processed_count/max(elapsed,1e-9) if processed_count else None,
                'work':work,'provenance':dict(self.provenance(),frame=index,view=mode,inference_device='cpu'),'state':'Ready',
                'label':'PROVISIONAL preview FPS includes sequential decoding and overlays; excludes setup/JPEG. Cached frames have no FPS.'}


class CancelledOperation(RuntimeError):
    pass
