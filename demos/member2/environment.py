"""Inspect the existing trusted local model/environment without installing or downloading."""
from __future__ import annotations
import importlib.metadata as md
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.member2.audit import digest
from soccertrack.detector_yolo import _offline


def capture(weights,expected,output=None):
    from datetime import datetime,timezone
    from demos.member2.safety import output_directory,require_absent
    out=output_directory(output if output is not None else ROOT/'outputs/member2_scientific/environment_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    require_absent(out,['architecture.json','environment.json'])
    weights=Path(weights)
    if not weights.is_absolute() or not weights.is_file() or digest(weights)!=expected:
        raise RuntimeError('approved absolute local checkpoint and SHA-256 required')
    _offline()
    import torch
    from ultralytics import YOLO
    net=YOLO(str(weights),task='detect').model
    arch={'parameters':sum(p.numel() for p in net.parameters()),
          'layers':[{'index':i,'type':type(layer).__name__,'parameters':sum(p.numel() for p in layer.parameters())} for i,layer in enumerate(net.model)],
          'yaml':net.yaml,'strides':net.stride.tolist(),'head':type(net.model[-1]).__name__,
          'nc':net.model[-1].nc,'reg_max':net.model[-1].reg_max,'weights_sha256':expected}
    env={'python':sys.version,'python_executable':sys.executable,
         'packages':{p:md.version(p) for p in ['ultralytics','torch','torchvision','numpy','pandas','opencv-python','lap','nbformat','nbclient','nbconvert','ipykernel','pytest']},
         'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'vram':torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
         'disk_free_D':shutil.disk_usage(ROOT).free,'disk_free_C':shutil.disk_usage('C:/').free}
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('architecture.json',arch),('environment.json',env)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    return arch,env

if __name__=='__main__':
    from demos.member2.scientific_workflow import WEIGHTS,EXPECTED
    arch,env=capture(WEIGHTS,EXPECTED)
    print(arch['parameters'],env['gpu'])
