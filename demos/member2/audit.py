"""Read-only preservation inventory; generated records stay under ignored outputs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/member2_scientific'
OWNED = {'src/soccertrack/detector_yolo.py', 'src/soccertrack/tracker_bytetrack.py',
         'scripts/run_yolo.py', 'configs/yolo.yaml', 'tests/test_detector_yolo.py',
         'tests/test_tracker_bytetrack.py', 'docs/phase2/sections/member_2_yolo.md'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def inventory():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').rstrip('\0').split('\0')
    paths = [ROOT / p for p in tracked if p not in OWNED and not p.startswith('demos/member2/')]
    for name in ('member1_scientific', 'member1_demo'):
        paths.extend(p for p in (ROOT / 'outputs' / name).rglob('*') if p.is_file())
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'preservation_before.json'
    if target.exists():
        before = json.loads(target.read_text())
        changes = [p for p, h in before.items() if not (ROOT / p).is_file() or digest(ROOT / p) != h]
        print(json.dumps({'protected_files': len(before), 'changed': changes}))
        if changes:
            raise SystemExit(1)
    else:
        data = inventory()
        target.write_text(json.dumps(data, indent=2))
        print(f'Inventoried {len(data)} protected files')
