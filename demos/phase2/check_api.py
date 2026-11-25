"""HTTP API acceptance, explicitly not a substitute claim for browser visual testing."""
from pathlib import Path
import argparse,base64,json,re,urllib.request,urllib.error
def check(run,port):
    base=f'http://127.0.0.1:{port}'
    page=urllib.request.urlopen(base,timeout=20).read().decode()
    token=re.search(r"window.DEMO_TOKEN='([^']+)'",page).group(1)
    def call(endpoint,body):
        req=urllib.request.Request(base+'/api/'+endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Demo-Token':token})
        return json.loads(urllib.request.urlopen(req,timeout=30).read())
    results=call('final/results',{});best=results['selected_scratch_detector'];cases=[]
    original=None
    for view,method in [('original',best),('gt',best),('pitch',best),('foreground',best),('proposals',best),('tiles','yolov8n_coco'),('detection',best),('detection','yolov8n_coco'),('tracks',best),('tracks','yolov8n_coco')]:
        value=call('final/frame',dict(index=0,view=view,detector=method,threshold=0 if view=='tracks' else .5))
        assert value['frame']==0 and value['config_sha256']==results['config_sha256']
        assert base64.b64decode(value['original']).startswith(b'\xff\xd8')
        if view=='original':original=value['original'];assert original==value['processed']
        if view=='gt':assert original==value['original'] and original!=value['processed']
        if view=='tiles':assert value['counts']['tiles']==4
        cases.append(dict(view=view,method=method,frame=0,scope=value['scope'],counts=value['counts']))
    for name in ('117092.mp4','128057.mp4','132831.mp4'):
        try:call('upload-check',dict(name=name,size=10));raise AssertionError('Protected filename accepted')
        except urllib.error.HTTPError as e:assert e.code==400
    report=dict(api_cases=cases,protected_uploads_rejected=True,browser_visual_testing='BLOCKED: ComputerUse could not verify current Windows browser URL; no bypass attempted',
        note='API/control-state/serialization verification only; play/pause/visual layout require manual browser acceptance')
    (run/'gui_api_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('API verification passed:',len(cases),'views; protected upload guards pass')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--port',type=int,default=8765);a=p.parse_args();check(a.run,a.port)
