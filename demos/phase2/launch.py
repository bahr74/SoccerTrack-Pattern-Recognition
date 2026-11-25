"""Launch the existing integrated application using a verified accepted run."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);p.add_argument('--port',type=int,default=8765);a=p.parse_args()
    run=a.run or ROOT/json.loads((ROOT/'outputs/phase2_final/latest.json').read_text())['path']
    run=run.resolve()
    if not run.is_relative_to(ROOT/'outputs/phase2_final') or not (run/'accepted_results.json').is_file():raise ValueError('Accepted final run required')
    subprocess.run([sys.executable,'-B',str(ROOT/'demos/member1/app.py'),'--final-run',str(run),'--port',str(a.port)],cwd=ROOT,check=True)
if __name__=='__main__':main()
