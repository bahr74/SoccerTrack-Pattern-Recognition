"""Qualitative tracking transitions from accepted tracks; never substituted for CLEAR IDSW."""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from demos.phase2.workflow import read,write
from soccertrack import config,data_io,contracts as C,metrics as M,visualize
def build(run):
    run=Path(run).resolve();data=read(run/'accepted_results.json');gt=data_io.load_gt_tracks('117093',config.load_data_sources())
    methods=[data['selected_scratch_detector']+'+iou_hungarian','yolov8n_coco+bytetrack'];fig,axes=plt.subplots(2,1,figsize=(12,7));events=[];picked=None
    for ax,method in zip(axes,methods):
        table=C.read_table(run/data['artifacts'][f'tracks/{method}/win60s']['path'],'tracks')
        rows=[];last={}
        for f in range(1500):
            truth=gt[gt.frame==f].reset_index(drop=True);pred=table[table.frame==f].reset_index(drop=True)
            result=M.match_frame(truth[['x','y','w','h']].to_numpy(),pred[['x','y','w','h']].to_numpy(),pred.score.to_numpy(),.5)
            for i,gi in enumerate(result['matched_gt']):
                if gi<0:continue
                gid=int(truth.iloc[gi].track_id);pid=int(pred.iloc[i].track_id);rows.append(dict(frame=f,gt_id=gid,pred_id=pid))
                if gid in last and last[gid][1]!=pid:
                    event=dict(pipeline=method,frame=f,gt_id=gid,previous_frame=last[gid][0],previous_pred_id=last[gid][1],pred_id=pid,diagnostic='greedy IoU matched-ID change; not official CLEAR IDSW')
                    events.append(event)
                    if picked is None and f>10 and f-last[gid][0]==1:picked=(event,table)
                last[gid]=(f,pid)
        d=pd.DataFrame(rows)
        changes={gid:sum((g.pred_id.diff().fillna(0)!=0)) for gid,g in d.groupby('gt_id')}
        for gid in sorted(changes,key=changes.get,reverse=True)[:3]:
            g=d[d.gt_id==gid];ax.scatter(g.frame/25,g.pred_id,s=4,label=f'GT{gid}')
        ax.set(xlabel='Time (seconds)',ylabel='Matched predicted ID',title=method+' — qualitative identity/fragmentation timeline');ax.legend()
    fig.tight_layout();fig.savefig(run/'figures/tracking_timeline.png',dpi=150);plt.close(fig)
    pd.DataFrame(events).to_csv(run/'tables/tracking_transition_diagnostics.csv',index=False)
    if picked:
        event,table=picked;f=event['frame'];clip=config.match_paths('117093',config.load_data_sources()).clip;images=[]
        for index,image in data_io.iter_frames(clip,[f-1,f]):
            pane=visualize.draw_boxes(image,table,index);cv2.putText(pane,f'{event["pipeline"]} | frame {index} | diagnostic GT{event["gt_id"]}: {event["previous_pred_id"]} -> {event["pred_id"]}',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(255,255,255),3);images.append(cv2.resize(pane,(2048,540)))
        cv2.imwrite(str(run/'figures/tracking_transition.png'),np.vstack(images));write(run/'tracking_example.json',event)
    else:raise RuntimeError('No unambiguous consecutive qualitative transition found; do not fabricate one')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();build(a.run)
