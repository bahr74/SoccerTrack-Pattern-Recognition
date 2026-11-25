"""Run bounded real production CLIs, retaining exact commands and output."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run=a.run.resolve()
    if not run.is_relative_to(ROOT/'outputs/phase2_final'):raise ValueError('Final run required')
    folder=run/'smoke';folder.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',MPLCONFIGDIR=str(run/'mpl'),YOLO_CONFIG_DIR=str(run/'yolo'))
    commands=[['scripts/check_data.py','--split','val','--no-hash'],
      ['scripts/run_scratch.py','score-patches','--detector','scratch_hog_svm','--split','val','--output',str(folder/'patches.csv')]]
    for method in ('scratch_mwb0','scratch_hog_svm','scratch_cnn'):
        commands.append(['scripts/run_scratch.py','detect','--detector',method,'--split','val','--match','117093','--frameset','smoke'])
    commands += [['scripts/run_yolo.py','detect','--detector','yolov8n_coco','--split','val','--match','117093','--frameset','smoke','--max-frames','2'],
      ['scripts/run_scratch.py','track','--detector','scratch_hog_svm','--split','val','--match','117093','--frameset','smoke'],
      ['scripts/run_evaluation.py','detection','--detector','scratch_hog_svm','--split','val','--match','117093','--frameset','smoke']]
    results=[]
    for i,args in enumerate(commands):
        print('smoke',i,args,flush=True)
        result=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
        (folder/f'command_{i}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        results.append(dict(command=args,returncode=result.returncode))
        (folder/'results.json').write_text(json.dumps(results,indent=2))
        if result.returncode:raise RuntimeError(f'Smoke{i} failed; inspect saved log')
if __name__=='__main__':main()
