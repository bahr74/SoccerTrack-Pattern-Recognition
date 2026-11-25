"""Native-resolution tiled COCO inference. Shared writers remain integration gates."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
from numbers import Integral
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
from .contracts import DETECTION_COLUMNS


def make_tiles(width: int, height: int, tile_w: int, tile_h: int, overlap: int) -> list[tuple[int,int,int,int]]:
    """Cover the image, with minimum overlap and flush final tiles."""
    for value in (width,height,tile_w,tile_h):
        if isinstance(value,bool) or not isinstance(value,Integral) or value <= 0:
            raise ValueError('image and tile dimensions must be positive integers')
    if isinstance(overlap,bool) or not isinstance(overlap,Integral) or not 0 <= overlap < min(tile_w,tile_h):
        raise ValueError('invalid overlap')
    def starts(length,size):
        return [0] if length <= size else sorted(set(list(range(0,length-size+1,size-overlap))+[length-size]))
    return [(x,y,min(x+tile_w,width),min(y+tile_h,height)) for y in starts(height,tile_h) for x in starts(width,tile_w)]


def _boxes(value):
    value=np.asarray(value,dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 4:
        raise ValueError('boxes must have shape (N,4)')
    return value.copy()


def tile_to_frame(boxes_xyxy: np.ndarray, tile: tuple[int,int,int,int]) -> np.ndarray:
    """Translate tile-local xyxy. Frame producer handles clipping."""
    boxes=_boxes(boxes_xyxy)
    if len(tile)!=4 or not np.isfinite(tile).all() or tile[2]<=tile[0] or tile[3]<=tile[1]:
        raise ValueError('invalid tile rectangle')
    return boxes+np.asarray([tile[0],tile[1],tile[0],tile[1]],np.float32)


def merge_tile_detections(boxes_xyxy: np.ndarray, scores: np.ndarray, iou: float) -> tuple[np.ndarray,np.ndarray]:
    """CPU torchvision NMS; discard nonfinite, degenerate and invalid-score rows."""
    import torch
    from torchvision.ops import nms
    boxes=_boxes(boxes_xyxy)
    scores=np.asarray(scores,dtype=np.float32)
    if scores.shape!=(len(boxes),) or not np.isfinite(iou) or not 0<=iou<=1:
        raise ValueError('invalid scores shape or IoU')
    valid=np.isfinite(boxes).all(axis=1)&np.isfinite(scores)&(scores>=0)&(scores<=1)&((boxes[:,2:]-boxes[:,:2])>1).all(axis=1)
    boxes,scores=boxes[valid],scores[valid]
    keep=nms(torch.from_numpy(boxes),torch.from_numpy(scores),float(iou)).numpy()
    return boxes[keep],scores[keep]


def apply_pitch_filter(dets: pd.DataFrame, mask_path: Any) -> tuple[pd.DataFrame,str]:
    """Filter bottom centres by binary native-resolution pitch mask; missing is off."""
    import cv2
    if mask_path is None or not Path(mask_path).is_file():
        return dets,'pitch_filter=off'
    mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.isin(mask,[0,255]).all():
        raise ValueError('mask must be a readable binary 0/255 image')
    expected=dets.attrs.get('image_size')
    if expected is not None and tuple(expected)!=(mask.shape[1],mask.shape[0]):
        raise ValueError('mask resolution differs from frame')
    points=dets[['x','y']].to_numpy(float)+dets[['w','h']].to_numpy(float)*[.5,1]
    inside=np.isfinite(points).all(axis=1)&(points>=0).all(axis=1)&(points<=[mask.shape[1],mask.shape[0]]).all(axis=1)
    keep=np.zeros(len(dets),bool)
    xy=np.minimum(np.floor(points[inside]).astype(int),[mask.shape[1]-1,mask.shape[0]-1])
    keep[inside]=mask[xy[:,1],xy[:,0]]==255
    return dets.loc[keep].copy(),'pitch_filter=on'


def _offline():
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[2]/'outputs/member2_scientific/ultralytics'))


class YoloTiledDetector:
    """Full-frame BGR uint8 input; low-confidence observations retained for tracking."""
    def __init__(self,cfg: dict[str,Any],weights_path: Any) -> None:
        path=Path(weights_path)
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError('existing absolute local weights path required; downloads disabled')
        self.cfg=cfg.get('yolo',cfg)
        self.weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        _offline()
        import torch
        from ultralytics import YOLO
        requested=self.cfg.get('device','cpu')
        self.device=requested if requested=='cpu' or torch.cuda.is_available() else 'cpu'
        self.model=YOLO(str(path),task='detect')
        self.last_debug={}
        self.last_timing={}

    def detect_frame(self,frame_bgr: np.ndarray,frame_id: int) -> pd.DataFrame:
        import torch
        if frame_bgr.dtype!=np.uint8 or frame_bgr.ndim!=3 or frame_bgr.shape[2]!=3 or not all(frame_bgr.shape[:2]):
            raise ValueError('expected nonempty BGR uint8 HxWx3')
        if isinstance(frame_id,bool) or not isinstance(frame_id,Integral) or frame_id<0:
            raise ValueError('frame_id must be nonnegative integer')
        cfg=self.cfg
        tiling,infer=cfg['tiling'],cfg['inference']
        floor=float(infer['conf_floor'])
        if not .05<=floor<=1 or cfg.get('classes',[0])!=[0]:
            raise ValueError('requires conf_floor >= .05 and classes=[0]')
        height,width=frame_bgr.shape[:2]
        def sync():
            if self.device!='cpu' and torch.cuda.is_available():
                torch.cuda.synchronize()
        sync()
        begin=time.perf_counter()
        tiles=make_tiles(width,height,tiling['tile_width'],tiling['tile_height'],tiling['overlap_px'])
        crops=[np.ascontiguousarray(frame_bgr[y0:y1,x0:x1]) for x0,y0,x1,y1 in tiles]
        prep=time.perf_counter()
        predictions=self.model.predict(crops,imgsz=tiling['imgsz'],conf=floor,iou=infer['iou_nms_in_tile'],classes=[0],
            device=self.device,half=cfg.get('half',False),max_det=infer['max_det_per_tile'],verbose=False,save=False)
        sync()
        predicted=time.perf_counter()
        if len(predictions)!=len(tiles):
            raise RuntimeError('tile result count mismatch')
        local,global_boxes,scores=[],[],[]
        for tile,result in zip(tiles,predictions):
            box=result.boxes.xyxy.cpu().numpy()
            score=result.boxes.conf.cpu().numpy()
            valid=(score>=floor)&(result.boxes.cls.cpu().numpy()==0)
            box,score=box[valid],score[valid]
            local.append(box.copy())
            global_boxes.append(tile_to_frame(box,tile))
            scores.append(score)
        boxes=np.concatenate(global_boxes)
        scores=np.concatenate(scores)
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width)
        boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
        before=boxes.copy()
        boxes,scores=merge_tile_detections(boxes,scores,cfg['merge']['global_nms_iou'])
        xywh=boxes.copy()
        xywh[:,2:]-=xywh[:,:2]
        table=pd.DataFrame(np.column_stack([np.full(len(boxes),frame_id),xywh,scores]),columns=DETECTION_COLUMNS)
        table['frame']=table['frame'].astype('int64')
        table.attrs['image_size']=(width,height)
        self.last_debug={'tiles':tiles,'local_boxes':local,'pre_nms':before,'post_nms':boxes.copy()}
        self.last_timing={'tile_preprocessing_seconds':prep-begin,'predict_seconds':predicted-prep,
                          'merge_seconds':time.perf_counter()-predicted,'processing_seconds':time.perf_counter()-begin}
        return table


def run_detection(detector_id: str,match_id: str,split: str,frameset: str,cfg: dict[str,Any],sources: dict[str,Any],frozen_test: bool=False) -> pd.DataFrame:
    """Official producer. Unavailable shared helpers fail without publishing provisional outputs."""
    from . import config,contracts,data_io
    from .split import require_test_access,split_of,log_test_access
    require_test_access(split,frozen_test)
    if split_of(match_id,cfg)!=split:
        raise ValueError('match does not belong to requested split')
    if detector_id!='yolov8n_coco':
        raise ValueError('only Phase 2 yolov8n_coco is implemented')
    if cfg['yolo']['inference']['conf_floor']!=cfg['evaluation']['score_floor']:
        raise ValueError('score_floor differs from evaluation config')
    paths=config.match_paths(match_id,sources)
    info=data_io.video_info(paths.clip)
    frames=config.frameset_range(frameset,info['frame_count_header'],cfg)
    path=config.artifact_path('detections',detector_id,split,match_id,frameset)
    limit=cfg.get('_member2_max_frames')
    if limit is not None:
        if isinstance(limit,bool) or not isinstance(limit,int) or limit<=0:
            raise ValueError('debug maximum must be a positive integer')
        frames=frames[:limit]
        path=config.repo_root()/'outputs/debug/detections'/detector_id/split/f'{match_id}__{frameset}.csv'
        path.parent.mkdir(parents=True,exist_ok=True)
    config.set_seeds(cfg['project']['seed'])
    detector=YoloTiledDetector(cfg,config.weights_path(cfg['yolo']['weights_key'],sources))
    tables,seen,processing=[],[],0.
    begin=time.perf_counter()
    iterator=data_io.iter_frames(paths.clip,frames)
    note='pitch_filter=off'
    try:
        for frame_id,image in iterator:
            table=detector.detect_frame(image,frame_id)
            processing+=detector.last_timing['processing_seconds']
            if cfg['yolo']['merge']['pitch_filter']:
                table,note=apply_pitch_filter(table,config.repo_root()/'outputs/masks'/f'{match_id}_pitch.png')
            tables.append(table)
            seen.append(frame_id)
    finally:
        iterator.close()
    elapsed=time.perf_counter()-begin
    if seen!=list(frames):
        raise RuntimeError('decoded coverage differs from requested frameset; artifact not published')
    result=pd.concat(tables,ignore_index=True) if tables else pd.DataFrame(columns=DETECTION_COLUMNS)
    meta=dict(schema_version=contracts.SCHEMA_VERSION,kind='detections',method=detector_id,tracker=None,split=split,
        match=match_id,frameset=frameset,frames={'start':frames.start,'stop':frames.stop,'stride':frames.step},
        image_width=info['width'],image_height=info['height'],fps=info['fps'],score_floor=cfg['evaluation']['score_floor'],
        operating_threshold=cfg['yolo']['operating_threshold'],frozen_test=frozen_test,config_sha256=config.config_sha256(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=config.repo_root(),text=True).strip(),
        created_utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),runtime_seconds=processing,
        frames_per_second=len(seen)/processing if processing else 0,device=str(detector.device),
        notes=f'{note}; weights_sha256={detector.weights_sha256}; decode_included_seconds={elapsed}; decode_included_fps={len(seen)/elapsed if elapsed else 0}; processing includes tile/predict/NMS; excludes model load, writing, overlays')
    contracts.validate_detections(result,meta)
    contracts.validate_run_meta(meta)
    contracts.write_table(result,path,'detections')
    contracts.write_run_meta(meta,path)
    if split=='test':
        log_test_access('Member 2',detector_id,[path.relative_to(config.repo_root()).as_posix()],{'yolov8n_coco':detector.weights_sha256})
    return result


def finetune(cfg: dict[str,Any],sources: dict[str,Any]) -> Any:
    """Phase 3 remains unauthorized."""
    raise NotImplementedError('TODO(Member 2, Phase 3)')
