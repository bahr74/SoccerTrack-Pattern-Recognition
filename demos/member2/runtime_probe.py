"""Small paired, warmed runtime probe; distinct from full-frameset processing FPS."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.member2.scientific_workflow import *

def probe(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text())
    if 'required_detection_results' not in manifest:
        raise RuntimeError('wait for full validation completion')
    ids=manifest['development_frame_ids'][:8]
    frames=list(iter_frames(CLIP,ids))
    rows=[]
    for name,cfg in manifest['candidates'].items():
        detector=YoloTiledDetector(cfg,WEIGHTS)
        detector.detect_frame(frames[0][1],frames[0][0])  # excluded setup/warmup
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        for frame,image in frames:
            detector.detect_frame(image,frame)
            rows.append({'configuration':name,'frame':frame,**detector.last_timing,
                'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})
        del detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(run/'warmed_runtime_probe.csv',index=False)
    write_json(run/'runtime_probe_definition.json',{'frame_ids':ids,'frames_per_candidate':8,
        'decode_included':False,'model_load_included':False,'first_warmup_included':False,
        'tile_preprocessing_included':True,'predict_includes':'Ultralytics preprocessing, model inference, local NMS',
        'global_merge_included':True,'cuda_synchronized':True,'writing_and_overlays_included':False,
        'note':'Small paired warmed probe; not full-run end-to-end throughput. Whole-run timings include cold starts; a separate GPU smoke test overlapped part of the required run, so those timings may contain resource contention.'})
    print(pd.DataFrame(rows).groupby('configuration').processing_seconds.agg(['mean','median']))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run')
    probe(parser.parse_args().run)
