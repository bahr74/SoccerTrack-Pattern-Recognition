"""Validate delivery scope and record final source/evidence hashes, without Git writes."""
from __future__ import annotations
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from demos.member2.audit import digest,OWNED
from demos.member2.write_handoff import read_log


def finalize():
    base=ROOT/'outputs/member2_scientific'
    run=base/'runs/member2_validation_20261125'
    before=json.loads((base/'preservation_before.json').read_text())
    changed=[p for p,h in before.items() if not (ROOT/p).is_file() or digest(ROOT/p)!=h]
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    assert branch=='feature/yolo-pipeline'
    diff=subprocess.check_output(['git','diff','--name-only','-z'],cwd=ROOT).decode().rstrip('\0').split('\0')
    assert set(diff)<=OWNED
    check=subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,text=True)
    assert check.returncode==0,check.stdout
    sources=[ROOT/p for p in OWNED]+[ROOT/'tests/test_member2_scientific.py']
    sources += [p for p in (ROOT/'demos/member2').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    assert all(p.suffix in ('.py','.md','.ipynb','.yaml') for p in sources)
    source_hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(set(sources))}
    artifact_hashes=json.loads((run/'artifact_hashes.json').read_text())
    assert all(digest(run/p)==h for p,h in artifact_hashes.items())
    m=json.loads((run/'manifest.json').read_text())
    assert all(digest(run/'source_snapshot'/p)==h for p,h in m['source_sha256'].items())
    summary={name:read_log(base/name).strip().splitlines()[-1] for name in
        ['baseline_pytest.txt','member2_unit_tests.txt','member2_gpu_tests.txt','member2_data_tests.txt','member2_all_tests.txt','full_pytest_final.txt']}
    evidence={p: digest(run/p) for p in ['manifest.json','artifact_hashes.json','notebook_execution.json',
        'member2_yolo_pipeline_experiments_executed.ipynb','member2_yolo_pipeline_experiments.html','annotated_validation.mp4']}
    result={'created_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'branch':branch,
        'git_commit':m['git_commit'],'status':'PARTIAL — SHARED DEPENDENCIES BLOCKED',
        'protected_files':len(before),'protected_files_changed':changed,'source_files':source_hashes,
        'run':run.relative_to(ROOT).as_posix(),'evidence_sha256':evidence,'tests':summary,
        'config_files_sha256':{name:digest(ROOT/'configs'/name) for name in ['project.yaml','scratch.yaml','yolo.yaml','evaluation.yaml']},
        'official_config_sha256':None,'official_config_hash_blocker':'Member 4 config.config_sha256 remains placeholder',
        'provenance_note':'The original experiment manifest and matching source_snapshot are authoritative for recorded numerical results. Later Member 2 documentation/reproduction helpers are independently hashed here. No old result is relabeled as a rerun.',
        'held_out_access':False,'downloads':False,'package_changes':False,'git_writes':False}
    (base/'delivery_manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    (base/'preservation_after.txt').write_text(json.dumps({'protected_files':len(before),'changed':changed})+'\n',encoding='utf-8')
    print(json.dumps({'branch':branch,'protected_files_unchanged':len(before),'reviewable_source_files':len(source_hashes),
                      'verified_stable_artifacts':len(artifact_hashes),'tests':summary},indent=2))

if __name__=='__main__':
    finalize()
