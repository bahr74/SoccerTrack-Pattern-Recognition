"""Bounded actual raw/morphology stage exports using selected shared scratch functions."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import cv2,numpy as np
from demos.phase2.workflow import read,copy_new,journal
from soccertrack import config,data_io,segmentation,proposals,detector_scratch as D,visualize
def build(run):
    cfg=config.load_config();sc=cfg['scratch'];clip=config.match_paths('117093',config.load_data_sources()).clip
    bg=cv2.imread(str(ROOT/sc['publication']['backgrounds']['117093']));pitch=segmentation.pitch_mask(bg,sc,'117093',None);size=proposals.SizeModel(**read(ROOT/sc['publication']['size_model']))
    folder=run/'stages';folder.mkdir(exist_ok=True)
    for f,image in data_io.iter_frames(clip,[0,500,1000]):
        pp,raw,clean=D.prepare_frame_proposals(image,f,bg,pitch,size,sc)
        for name,value in [('frame',image),('foreground',raw.astype(np.uint8)*255),('morph',clean.astype(np.uint8)*255),('proposals',visualize.draw_boxes(image,pp.drop(columns='score'),f))]:
            path=folder/f'{f:06d}_{name}.png';cv2.imwrite(str(path),value)
            copy_new(path,ROOT/'outputs/stages/117093'/path.name)
    journal(run,'Bounded three-frame stage export distinguishes raw foreground and cleaned morphology; no classifier inference')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run.resolve())
