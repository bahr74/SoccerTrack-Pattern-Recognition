r"""Member 1 loopback application. Launch .venv\Scripts\python.exe demos/member1/app.py."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
import argparse,base64,json,mimetypes,secrets,socket,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
import cv2
from experiment import ROOT,OUT,safe_video,export_preview
from soccertrack import data_io
from preview import Preview,CancelledOperation

MAX_UPLOAD=512*1024*1024

class UploadTooLarge(ValueError):pass

def validate_upload(name,length):
    if any(c in name for c in ('/','\\',':','\x00')) or Path(name).name!=name or Path(name).suffix.lower() not in ('.avi','.mp4','.mov','.mkv'):
        raise ValueError('Select an MP4, AVI, MOV or MKV video file')
    if Path(name).stem in ('128057','132831'):raise PermissionError('Held-out test matches are prohibited')
    if length>MAX_UPLOAD:raise UploadTooLarge('Video exceeds 512 MiB. Choose a smaller file or use Load Real Validation Demo, which reads the approved clip directly.')
    if length<=0:raise ValueError('The selected video is empty')


def make_server(port=8765):
    import torch,scientific_workflow as w
    torch.set_num_threads(2);token=secrets.token_urlsafe(24);preview=Preview()
    log_path=w.BASE/'corrections/runtime'/('operations_'+secrets.token_hex(8)+'.jsonl');log_path.parent.mkdir(parents=True,exist_ok=True);log_lock=threading.Lock()
    def log(operation,status,**details):
        with log_lock:
            if log_path.exists() and log_path.stat().st_size>16*1024*1024:return
            with log_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),operation=operation,status=status,**details),allow_nan=False)+'\n')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data,kind='application/json'):
            if kind=='application/json':data=json.dumps(data,allow_nan=False).encode()
            try:
                self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data);self.wfile.flush()
            except (ConnectionError,OSError):log(urlparse(self.path).path,'Cancelled',reason='Client disconnected before the response was consumed')
        def local(self):return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}')
        def reject(self,status,error,length=0):
            state='Cancelled' if isinstance(error,CancelledOperation) else 'Error';log(urlparse(self.path).path,state,error=str(error))
            self.reply(status,{'error':str(error),'type':type(error).__name__,'state':state})
            # Reply first, then consume only a bounded amount already in flight.
            # Correct clients preflight size; never drain an unbounded rejected file.
            old_timeout=self.connection.gettimeout();deadline=time.monotonic()+2;remaining=min(max(length,0),8*1024*1024)
            try:
                self.connection.settimeout(.25)
                while remaining and time.monotonic()<deadline:
                    chunk=self.rfile.read1(min(65536,remaining))
                    if not chunk:break
                    remaining-=len(chunk)
            except (OSError,socket.timeout):pass
            finally:self.connection.settimeout(old_timeout);self.close_connection=True
        def do_GET(self):
            if not self.local():return self.reply(403,{'error':'Local host required','state':'Error'})
            p=urlparse(self.path);args=parse_qs(p.query)
            if p.path=='/':return self.reply(200,(Path(__file__).parent/'web/index.html').read_text(encoding='utf-8').replace('__TOKEN__',token).encode(),'text/html; charset=utf-8')
            if p.path in ('/app.js','/app.css'):
                path=Path(__file__).parent/'web'/p.path[1:];return self.reply(200,path.read_bytes(),'text/javascript; charset=utf-8' if path.suffix=='.js' else 'text/css; charset=utf-8')
            if p.path in ('/download','/artifact'):
                if args.get('token',[''])[0]!=token:return self.reply(403,{'error':'Invalid token','state':'Error'})
                folder=OUT if p.path=='/download' else w.BASE;path=(folder/args.get('file',[''])[0]).resolve()
                if not path.is_relative_to(folder.resolve()) or not path.is_file():return self.reply(404,{'error':'Artifact not found','state':'Error'})
                return self.reply(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.reply(404,{'error':'Unknown endpoint','state':'Error'})
        def do_POST(self):
            length=int(self.headers.get('Content-Length','0')) if self.headers.get('Content-Length','0').isdigit() else 0
            if not self.local():return self.reject(403,PermissionError('Local host required'),length)
            if self.headers.get('X-Demo-Token')!=token:return self.reject(403,PermissionError('Invalid token'),length)
            endpoint=urlparse(self.path).path;started=time.perf_counter();consumed=False
            try:
                if endpoint=='/api/upload':
                    name=parse_qs(urlparse(self.path).query).get('name',[''])[0];validate_upload(name,length)
                    folder=w.BASE/'uploads'/secrets.token_hex(8);folder.mkdir(parents=True);path=folder/name
                    with path.open('wb') as output:
                        remaining=length
                        while remaining:
                            chunk=self.rfile.read(min(1024*1024,remaining))
                            if not chunk:raise ValueError('Video upload did not finish; select the file again')
                            output.write(chunk);remaining-=len(chunk)
                    consumed=True;info=data_io.video_info(path)
                    image=next(data_io.iter_frames(path,[0]),None)
                    if image is None:raise ValueError('Video has metadata but no decodable first frame')
                    result={'video':str(path),'name':name,'state':'Ready','bytes':length}
                else:
                    if not 0<length<=32768:raise ValueError('Invalid JSON request size')
                    data=json.loads(self.rfile.read(length));consumed=True
                    if endpoint=='/api/upload-check':
                        validate_upload(data['name'],int(data['size']));result={'state':'Ready','max_upload_bytes':MAX_UPLOAD}
                    elif endpoint=='/api/cancel':
                        preview.cancel(data['operation_id']);result={'state':'Cancelled'}
                    else:
                        with preview.lock:result=self.perform(endpoint,data)
                log(endpoint,'Ready' if result.get('state')!='Cancelled' else 'Cancelled',seconds=time.perf_counter()-started,frame=result.get('frame'),provenance=result.get('provenance'))
                self.reply(200,dict(result,state=result.get('state','Ready')))
            except Exception as error:
                status=413 if isinstance(error,UploadTooLarge) else 409 if isinstance(error,CancelledOperation) else 400
                self.reject(status,error,0 if consumed else length)
        def perform(self,endpoint,data):
            source=data.get('source','synthetic');method=data.get('method','rules');operation=data.get('operation_id')
            if endpoint=='/api/dashboard':
                import ui_support
                return ui_support.dashboard()
            if endpoint=='/api/real':
                source=data.get('source','expanded');selection=preview.resolve(source,method);registry=preview._json(w.BASE/'local_sources.json');match=selection['cfg']['project']['split']['val'][0]
                item=registry['matches'][match];safe_video(item['clip']);safe_video(item['mot'])
                run=w.active_run();manifest=preview._json(run/'manifest.json');evidence=next(v for v in manifest['dataset'] if v['match']==match)
                bg=cv2.imread(str(ROOT/evidence['background']))
                if bg is None:raise FileNotFoundError('Cached validation background unavailable; rebuild explicitly before using the demo')
                return preview.open(item['clip'],item['mot'],evidence['video_header']['frame_count_header'],bg,source,method,operation)
            if endpoint=='/api/open':return preview.open(data['video'],data.get('gt') or None,data.get('background_window',60),source=source,method=method,operation_id=operation)
            if endpoint=='/api/frame':return preview.frame(int(data['index']),data.get('mode','rules'),source,method,int(data.get('width',1000)),operation)
            if endpoint=='/api/save-frame':
                frame=preview.frame(int(data['index']),data.get('mode','rules'),source,method,1280,operation)
                folder=w.BASE/'exports';folder.mkdir(parents=True,exist_ok=True);path=folder/(secrets.token_hex(8)+'.jpg');path.write_bytes(base64.b64decode(frame['processed']))
                w.write_json(path.with_suffix('.json'),{'status':'PROVISIONAL annotated frame','provenance':frame['provenance']})
                return {'artifact':path.relative_to(w.BASE).as_posix(),'provenance':frame['provenance']}
            if endpoint=='/api/export':
                if preview.path is None:raise ValueError('Open a video first')
                preview.begin_operation(operation);selection=preview.activate(preview.resolve(source,method));preview.pitch_for()
                folder=OUT/'exports'/secrets.token_hex(6);provenance=dict(preview.provenance(),view='tracks',inference_device='cpu')
                result=export_preview(preview.path,preview.gt,preview.cfg,selection['size'],folder,preview._model(source,method),min(30,preview.total),preview.background_window,preview.bg,cancel_check=preview.check_cancelled,provenance=provenance)
                result.update(file=(folder/'annotated_provisional.mp4').relative_to(OUT).as_posix(),provenance=provenance,predictions=(folder/'detections_provisional.csv').relative_to(OUT).as_posix(),tracks_file=(folder/'tracks_provisional.csv').relative_to(OUT).as_posix())
                return result
            if endpoint=='/api/browse':
                if str(data['folder']).startswith(('\\\\','//')):raise PermissionError('Local folders only')
                folder=Path(data['folder']).resolve()
                if any(part in ('128057','132831') for part in folder.parts):raise PermissionError('Held-out folder')
                files=[{'name':'..','path':str(folder.parent),'directory':True}]
                for p in sorted(folder.iterdir(),key=lambda v:v.name.lower()):
                    if p.stem in ('128057','132831'):continue
                    if p.is_dir() or p.suffix.lower() in ('.avi','.mp4','.mov','.mkv'):files.append({'name':p.name,'path':str(p),'directory':p.is_dir()})
                    if len(files)>=200:break
                return {'files':files}
            raise ValueError('Unknown endpoint')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.preview=preview;server.demo_token=token;server.operation_log=log_path
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8765);args=parser.parse_args();server=make_server(args.port)
    print(f'Member 1 provisional GUI: http://127.0.0.1:{server.server_port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.preview.close();server.server_close()

if __name__=='__main__':main()
