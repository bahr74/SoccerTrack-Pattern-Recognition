"""Integrated accepted validation provider; original/GT require no detector inference."""
from pathlib import Path
from collections import OrderedDict
import base64,json
import cv2
import numpy as np
from soccertrack import config,contracts as C,data_io,metrics as M

class FinalProvider:
    methods=('scratch_mwb0','scratch_hog_svm','scratch_cnn','yolov8n_coco')
    def __init__(self,run):
        self.run=Path(run).resolve();root=config.repo_root()
        if not self.run.is_relative_to(root/'outputs/phase2_final'):raise ValueError('Final output directory required')
        self.results=json.loads((self.run/'accepted_results.json').read_text())
        if self.results['config_sha256']!=config.config_sha256():raise ValueError('Accepted configuration no longer matches; do not mix results')
        self.cfg=config.load_config();self.paths=config.match_paths('117093',config.load_data_sources())
        self.gt=data_io.load_gt_tracks('117093',config.load_data_sources());self.tables={};self.images=OrderedDict();self.cap=None;self.position=-1
    def close(self):
        if self.cap is not None:self.cap.release()
    def table(self,key):
        if key not in self.results['artifacts']:raise ValueError('This combination/frame window has no accepted artifact')
        if key not in self.tables:
            import hashlib
            item=self.results['artifacts'][key];path=self.run/item['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Accepted table changed')
            self.tables[key]=C.read_table(path,key.split('/')[0])
        return self.tables[key]
    def original(self,index):
        if index in self.images:return self.images[index].copy()
        if self.cap is None:self.cap=cv2.VideoCapture(str(self.paths.clip))
        if self.position!=index-1:self.cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,image=self.cap.read()
        if not ok or int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))!=index+1:raise ValueError('Exact requested frame could not be decoded')
        self.position=index;self.images[index]=image
        while len(self.images)>16:self.images.popitem(last=False)
        return image.copy()
    def frame(self,index,method='scratch_hog_svm',view='detection',threshold=.5,width=1280):
        if not 0<=index<5976 or method not in self.methods or not 0<=threshold<=1 or not 320<=width<=1920:raise ValueError('Invalid analyzer selection')
        from soccertrack.visualize import draw_boxes
        image=self.original(index);canvas=image.copy();gt=self.gt[self.gt.frame==index];counts={};scope='cached canonical predictions'
        if view=='original':scope='decoded original; no inference'
        elif view=='gt':canvas=draw_boxes(image,gt,index);scope='ground truth; no inference'
        elif view=='tiles':
            from soccertrack.detector_yolo import make_tiles
            tiling=self.cfg['yolo']['tiling']
            boxes=make_tiles(4096,1080,tiling['tile_width'],tiling['tile_height'],tiling['overlap_px'])
            for n,(x1,y1,x2,y2) in enumerate(boxes):
                cv2.rectangle(canvas,(x1,y1),(x2-1,y2-1),(255,120,0),3)
                cv2.putText(canvas,f'Tile{n}:x={x1}..{x2}',(x1+12,70+35*n),cv2.FONT_HERSHEY_SIMPLEX,1.,(255,255,255),3)
            counts['tiles']=len(boxes);scope='selected native tile geometry; no inference'
        elif view in ('background','pitch','foreground','proposals'):
            from soccertrack import segmentation,proposals,detector_scratch as D
            bg=cv2.imread(str(config.repo_root()/self.cfg['scratch']['publication']['backgrounds']['117093']))
            mask=segmentation.pitch_mask(bg,self.cfg['scratch'],'117093',None)
            if view=='background':canvas=cv2.resize(bg,(4096,1080))
            elif view=='pitch':canvas=cv2.cvtColor(cv2.resize(mask.astype(np.uint8)*255,(4096,1080),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
            else:
                size=proposals.SizeModel(**json.loads((config.repo_root()/self.cfg['scratch']['publication']['size_model']).read_text()))
                p,_,fg=D.prepare_frame_proposals(image,index,bg,mask,size,self.cfg['scratch'])
                canvas=draw_boxes(image,p,index) if view=='proposals' else cv2.cvtColor(cv2.resize(fg.astype(np.uint8)*255,(4096,1080),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
                counts['proposals']=len(p)
            scope='bounded selected scratch preprocessing; no classifier/tracker inference'
        else:
            if index>=1500 and index%10:raise ValueError('No cached prediction for this frame; use a stride10 frame beyond1499')
            frameset='win60s' if index<1500 else 'eval10'
            if view=='tracks':
                if index>=1500:raise ValueError('Accepted tracking covers frames0–1499')
                tracker='bytetrack' if method=='yolov8n_coco' else 'iou_hungarian'
                key=f'tracks/{method}+{tracker}/win60s'
            else:key=f'detections/{method}/{frameset}'
            table=self.table(key);pred=table[(table.frame==index)&(table.score>=threshold)]
            canvas=draw_boxes(image,pred,index)
            r=M.match_frame(gt[['x','y','w','h']].to_numpy(),pred[['x','y','w','h']].to_numpy(),pred.score.to_numpy(),.5)
            counts.update(detections=len(pred),tp=int(r['tp_flags'].sum()),fp=int((~r['tp_flags']).sum()),fn=int((~r['gt_matched']).sum()))
            if view=='tracks':counts['visible_ids']=int(pred.track_id.nunique())
        def encode(bgr):
            resized=cv2.resize(bgr,(width,round(bgr.shape[0]*width/bgr.shape[1])),interpolation=cv2.INTER_AREA)
            return base64.b64encode(cv2.imencode('.jpg',resized,[cv2.IMWRITE_JPEG_QUALITY,91])[1]).decode()
        return dict(frame=index,original=encode(image),processed=encode(canvas),counts=counts,
                    method=method,view=view,threshold=threshold,config_sha256=self.results['config_sha256'],scope=scope,
                    official_threshold=0. if view=='tracks' else .5,interactive_settings_do_not_change_results=True,native_size=[4096,1080])
