"""Build native-aspect figures, an annotated video, and provenance from cached predictions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.member2.scientific_workflow import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def refresh_hashes(run):
    """Hash stable evidence, excluding derivative notebooks that verify this index."""
    from demos.member2.safety import require_absent
    run=require_absent(run,['artifact_hashes.json'])
    excluded={'artifact_hashes.json','notebook_execution.json'}
    hashes={p.relative_to(run).as_posix():digest(p) for p in run.rglob('*')
            if p.is_file() and p.name not in excluded and p.suffix not in ('.ipynb','.html')}
    write_json(run/'artifact_hashes.json',hashes)
    return hashes


def build(run):
    from demos.member2.safety import require_absent
    run=require_absent(run,['figures','annotated_validation.mp4','reference_threshold_comparison.csv',
                            'candidate_optima.csv','video_verification.json','artifact_hashes.json'])
    manifest=json.loads((run/'manifest.json').read_text())
    if not all((run/name).is_file() for name in ('architecture.json','environment.json','warmed_runtime_probe.csv')):
        raise RuntimeError('Run-local environment/architecture and runtime_probe are required before building derivatives')
    if 'required_detection_results' not in manifest:
        raise RuntimeError('required validation run is not complete')
    snapshot=run/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for relative,expected in manifest['source_sha256'].items():
        source=ROOT/relative
        target=snapshot/relative
        if not target.exists():
            if digest(source)!=expected:
                raise RuntimeError(f'Cannot archive changed experiment source: {relative}')
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(source.read_bytes())
    figures=run/'figures'
    figures.mkdir(exist_ok=True)
    dest=run/'required_validation'
    dets=pd.read_csv(dest/'detections.csv')
    tracks=pd.read_csv(dest/'tracks_starting.csv')
    gt,issues=clean_mot_geometry(read_mot_raw(MOT),(4096,1080),22)
    grid=pd.read_csv(run/'development_results.csv')
    threshold=manifest['selected_configuration']['operating_threshold']
    chosen=manifest['selected_candidate']
    def save(name):
        plt.tight_layout()
        plt.savefig(figures/name,dpi=150,bbox_inches='tight')
        plt.close()
    # Controlled candidate comparisons at the same score threshold, alongside their individual optima.
    at40=grid[np.isclose(grid.threshold,.4)].copy()
    best=grid.sort_values('f1',ascending=False).groupby('configuration',sort=False).head(1)
    at40[['configuration','threshold','tp','fp','fn','precision','recall','f1','processing_seconds','fps','tiles_per_frame']].to_csv(run/'reference_threshold_comparison.csv',index=False)
    best.to_csv(run/'candidate_optima.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in grid.groupby('configuration',sort=False):
        axes[0].plot(g.threshold,g.f1,marker='.',label=name)
    axes[0].set(xlabel='Operating score threshold',ylabel='Local F1 at IoU 0.5',ylim=(0,1),title='33 paired development frames')
    axes[0].legend(fontsize=8)
    warmed=pd.read_csv(run/'warmed_runtime_probe.csv').groupby('configuration').processing_seconds.mean()
    axes[1].bar(at40.configuration,[1/warmed[name] for name in at40.configuration],color='#237a9c')
    axes[1].tick_params(axis='x',rotation=20)
    axes[1].set(ylabel='Processing frames/s',title='Paired eight-frame warmed probe; decode excluded')
    fig.suptitle('PROVISIONAL local diagnostics; not official E3')
    save('controlled_experiments.png')
    curves=pd.read_csv(dest/'eval10_threshold_analysis.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for column in ['precision','recall','f1']:
        axes[0].plot(curves.threshold,curves[column],label=column)
    axes[0].axvline(threshold,linestyle='--',color='black',label='development-selected threshold')
    axes[0].set(xlabel='Operating threshold',ylabel='Fraction',ylim=(0,1))
    axes[0].legend()
    axes[1].plot(curves.recall,curves.precision,marker='.')
    axes[1].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Sampled score-threshold PR curve; no AP claimed')
    fig.suptitle('eval10: 598 validation frames, provisional local diagnostics')
    save('threshold_analysis.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    lengths=tracks.groupby('track_id').size()
    axes[0].hist(lengths,bins=30,color='#237a9c')
    axes[0].set(xlabel='Observations per predicted ID',ylabel='Count of IDs',title='Track lengths; gaps are not filled')
    diagnostics=manifest['tracking_diagnostics']
    axes[1].bar([r['configuration'] for r in diagnostics],[r['ids'] for r in diagnostics],color='#ba7231')
    axes[1].set(ylabel='Predicted IDs in 1,500 frames',title='Local fragmentation proxy; not official ID switches')
    save('tracking_diagnostics.png')
    gtframes={int(f):g for f,g in gt.groupby('frame')}
    _,labels=metrics(dets,gt,[0,500,1000],threshold)
    labelmap={f:{name:[] for name in ['tp','fp','fn']} for f in [0,500,1000]}
    for f,name,box in labels:
        labelmap[f][name].append(box)
    writer=cv2.VideoWriter(str(run/'annotated_validation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(2048,540))
    if not writer.isOpened():
        raise RuntimeError('video writer unavailable')
    wanted=sorted(set(list(range(150))+[500,1000]))
    generator=iter_frames(CLIP,wanted)
    try:
        for f,image in generator:
            if f in labelmap:
                overlay=image.copy()
                for label,color in [('tp',(0,255,0)),('fp',(0,0,255)),('fn',(255,0,0))]:
                    overlay=draw(overlay,labelmap[f][label],color)
                cv2.putText(overlay,f'Frame {f}: local IoU .5 TP green, FP red, FN blue; threshold {threshold:.2f}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                cv2.imwrite(str(figures/f'errors_frame_{f}.jpg'),overlay)
                # Native-pixel diagnostic crops, with their global origins recorded.
                crops=[]
                cropmeta=[]
                for label in ['fp','fn']:
                    for box in labelmap[f][label][:6]:
                        x,y,w,h=box
                        x0,y0=max(0,int(x)-40),max(0,int(y)-40)
                        x1,y1=min(4096,int(x+w)+40),min(1080,int(y+h)+40)
                        crop=image[y0:y1,x0:x1]
                        target=figures/f'failure_{f}_{label}_{len(crops)}.jpg'
                        cv2.imwrite(str(target),crop)
                        cropmeta.append({'file':target.name,'frame':f,'label':label,'box':box,'crop_origin':[x0,y0],'classification':'unreviewed visual example; no causal label inferred'})
                        crops.append(crop)
                write_json(figures/f'failure_{f}.json',cropmeta)
            if f==0:
                cv2.imwrite(str(figures/'gt_frame_0.jpg'),draw(image,gtframes[0][['x','y','w','h']].to_numpy(),(0,255,0)))
                # Show a genuine overlap seam at native scale.
                geometry=json.loads((dest/'stages/geometry.json').read_text())
                seam=geometry['tiles'][1][0]
                cv2.imwrite(str(figures/'overlap_crop.jpg'),image[250:650,max(0,seam-100):seam+180])
                fig,ax=plt.subplots(figsize=(15,4))
                ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
                for tid,group in tracks.groupby('track_id'):
                    if len(group)>=150:
                        ax.plot(group.x+group.w/2,group.y+group.h,linewidth=.8,alpha=.8)
                ax.set(xlim=(0,4096),ylim=(1080,0),title='Predicted bottom-centre trajectories, IDs with >=150 observations; pixel coordinates')
                ax.set_aspect('equal')
                save('trajectories.png')
            if f<150:
                subset=tracks[tracks.frame==f]
                overlay=draw(image,subset[['x','y','w','h']].to_numpy(),(255,0,255),
                             [f'ID {r.track_id} {r.score:.2f}' for r in subset.itertuples()])
                cv2.putText(overlay,f'yolov8n_coco + bytetrack | validation 117093 | frame {f} | provisional',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                writer.write(cv2.resize(overlay,(2048,540),interpolation=cv2.INTER_AREA))
    finally:
        generator.close()
        writer.release()
    capture=cv2.VideoCapture(str(run/'annotated_validation.mp4'))
    decoded=0
    try:
        while True:
            ok,frame=capture.read()
            if not ok:
                break
            assert frame.shape[:2]==(540,2048)
            decoded+=1
    finally:
        capture.release()
    if decoded!=150:
        raise RuntimeError(f'exported video incomplete: {decoded}/150 frames')
    write_json(run/'video_verification.json',{'decoded_frames':decoded,'fps':25,'source_frame_ids':list(range(150)),
        'native_size':[4096,1080],'export_size':[2048,540],'aspect_preserved':True,'status':STATUS,
        'tracks_source':'required_validation/tracks_starting.csv','detector':'yolov8n_coco','tracker':'bytetrack'})
    if not all((run/name).is_file() for name in ('architecture.json','environment.json')):
        raise RuntimeError('Run-local architecture/environment provenance is required')
    refresh_hashes(run)
    print('Built figures and verified 150-frame video:',run)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    build(parser.parse_args().run)
