"""Genuine headless Edge input/file-selection verification via standard-library CDP.

No browser packages are installed. Uses an isolated ignored browser profile.
"""
from pathlib import Path
import base64, hashlib, json, os, socket, struct, subprocess, threading, time
from urllib.request import urlopen
from urllib.parse import urlsplit
import app
import numpy as np, pandas as pd
from io import StringIO
from experiment import OUT,make_synthetic
from scientific_workflow import BASE,write_json

class CDP:
    def __init__(self,url):
        u=urlsplit(url); self.sock=socket.create_connection((u.hostname,u.port),timeout=30); self.sock.settimeout(40); self.serial=0
        key=base64.b64encode(os.urandom(16)).decode(); self.sock.sendall((f'GET {u.path} HTTP/1.1\r\nHost: {u.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n').encode())
        header=b''
        while not header.endswith(b'\r\n\r\n'): header+=self.sock.recv(1)
        assert header.startswith(b'HTTP/1.1 101 '), header
        expected=base64.b64encode(hashlib.sha1((key+'258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest())
        assert expected in header, 'Invalid WebSocket handshake accept key'
    def read(self,n):
        value=b''
        while len(value)<n:
            part=self.sock.recv(n-len(value))
            if not part: raise ConnectionError('Browser socket closed')
            value+=part
        return value
    def command(self,method,params=None):
        self.serial+=1; payload=json.dumps({'id':self.serial,'method':method,'params':params or {}}).encode(); key=os.urandom(4); n=len(payload)
        header=bytes([0x81,0x80|n]) if n<126 else bytes([0x81,0x80|126])+struct.pack('!H',n) if n<65536 else bytes([0x81,0x80|127])+struct.pack('!Q',n)
        self.sock.sendall(header+key+bytes(v^key[i%4] for i,v in enumerate(payload)))
        chunks=[]
        while True:
            first,second=self.read(2); length=second&127
            if length==126:length=struct.unpack('!H',self.read(2))[0]
            elif length==127:length=struct.unpack('!Q',self.read(8))[0]
            mask=self.read(4) if second&128 else None; data=self.read(length)
            if mask:data=bytes(v^mask[i%4] for i,v in enumerate(data))
            if first&15==8:raise ConnectionError('Browser closed CDP')
            if first&15 in (0,1): chunks.append(data)
            else:continue
            if not first&128:continue
            value=json.loads(b''.join(chunks)); chunks=[]
            if value.get('id')==self.serial:
                if 'error' in value:raise RuntimeError(value['error'])
                return value.get('result',{})
    def js(self,expression):
        value=self.command('Runtime.evaluate',{'expression':expression,'returnByValue':True,'awaitPromise':True})
        if value.get('exceptionDetails'):raise RuntimeError(value['exceptionDetails'])
        return value.get('result',{}).get('value')
    def wait(self,expression,timeout=55):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if self.js('Boolean('+expression+')'):return
            time.sleep(.15)
        raise TimeoutError(expression+'; '+str(self.js('document.body.innerText.slice(-2000)')))
    def click(self,selector):
        self.js(f'document.querySelector({json.dumps(selector)}).scrollIntoView({{block:"center"}})')
        point=self.js(f'(()=>{{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}}}})()')
        for kind in ['mousePressed','mouseReleased']:self.command('Input.dispatchMouseEvent',dict(type=kind,button='left',clickCount=1,**point))
    def key(self,key):
        for kind in ['keyDown','keyUp']:self.command('Input.dispatchKeyEvent',{'type':kind,'key':key,'code':key,'windowsVirtualKeyCode':{'ArrowUp':38,'ArrowDown':40,'ArrowRight':39,'Home':36,'End':35,'Enter':13}.get(key,0)})

def main():
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    if not edge.is_file():raise FileNotFoundError('Edge not installed')
    folder=BASE/'browser_verification'/str(time.time_ns()); folder.mkdir(parents=True)
    profile=folder/'profile'; source=folder/'browser_fixture.avi'; make_synthetic(source,12)
    server=app.make_server(0); worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    base=f'http://127.0.0.1:{server.server_port}'; process=subprocess.Popen([str(edge),'--headless=new','--no-sandbox','--disable-gpu','--in-process-gpu','--no-first-run','--remote-debugging-port=0','--remote-allow-origins=*',f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    checks=[]
    def passed(name):checks.append(name);print('PASS '+name,flush=True)
    cdp=None
    try:
        for _ in range(200):
            if (profile/'DevToolsActivePort').is_file():break
            time.sleep(.1)
        port=int((profile/'DevToolsActivePort').read_text().splitlines()[0])
        tabs=json.load(urlopen(f'http://127.0.0.1:{port}/json/list')); target=next(t for t in tabs if t['type']=='page'); cdp=CDP(target['webSocketDebuggerUrl'])
        cdp.command('Page.enable');cdp.command('Runtime.enable')
        cdp.command('Emulation.setDeviceMetricsOverride',{'width':1440,'height':1000,'deviceScaleFactor':1,'mobile':False})
        cdp.command('Page.navigate',{'url':base})
        cdp.wait("document.querySelector('#file') && document.querySelector('#patchTable table')")
        passed('Four-view page and real experiment dashboard loaded')
        import ctypes,msvcrt
        large=folder/'oversize_fixture.mp4'
        with large.open('wb') as stream:
            returned=ctypes.c_ulong();ok=ctypes.windll.kernel32.DeviceIoControl(ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno())),0x900c4,None,0,None,0,ctypes.byref(returned),None)
            if not ok:raise OSError('Could not mark the oversize browser fixture sparse')
            stream.truncate(app.MAX_UPLOAD+1)
        doc=cdp.command('DOM.getDocument');node=cdp.command('DOM.querySelector',{'nodeId':doc['root']['nodeId'],'selector':'#file'})
        before=cdp.js("performance.getEntriesByType('resource').filter(e=>e.name.includes('/api/upload')).length")
        cdp.command('DOM.setFileInputFiles',{'nodeId':node['nodeId'],'files':[str(large)]})
        cdp.wait("document.body.dataset.state==='Error' && document.querySelector('#error').textContent.includes('512 MiB')")
        assert before==cdp.js("performance.getEntriesByType('resource').filter(e=>e.name.includes('/api/upload')).length");passed('Oversize browser file rejected before any HTTP upload')
        doc=cdp.command('DOM.getDocument'); node=cdp.command('DOM.querySelector',{'nodeId':doc['root']['nodeId'],'selector':'#file'})
        cdp.command('DOM.setFileInputFiles',{'nodeId':node['nodeId'],'files':[str(source)]})
        cdp.wait("document.querySelector('#original').src.startsWith('data:image') && document.querySelector('#loading').textContent==='Ready'")
        assert cdp.js("document.body.dataset.state")== 'Ready'
        assert 'browser_fixture.avi' in cdp.js("document.querySelector('#metadata').innerText");passed('Browser file input uploaded and decoded actual selected video')
        cdp.click('#next');cdp.wait("document.querySelector('#position').textContent.startsWith('Frame 1 /')");passed('Next frame control')
        cdp.click('#previous');cdp.wait("document.querySelector('#position').textContent.startsWith('Frame 0 /')");passed('Previous frame control')
        cdp.click('#play');cdp.wait("!document.querySelector('#position').textContent.startsWith('Frame 0 /')");cdp.click('#play');cdp.wait("document.body.dataset.state==='Ready'");paused=cdp.js("document.querySelector('#position').textContent");time.sleep(.4);assert paused==cdp.js("document.querySelector('#position').textContent");assert cdp.js("document.body.dataset.playing")== 'false';passed('Pause stops frame advancement')
        cdp.click('#seek');cdp.key('Home')
        for _ in range(5):cdp.key('ArrowRight')
        cdp.wait("document.querySelector('#position').textContent.startsWith('Frame 5 /') && document.body.dataset.state==='Ready'");assert cdp.js("document.querySelector('#processed').dataset.frame")== '5';passed('Seek reaches requested zero-based frame 5')
        cdp.click('#real');cdp.wait("document.querySelector('#metadata').innerText.includes('117093.mp4') && document.querySelector('#loading').textContent==='Ready'");passed('Approved real-validation demo loading')
        cdp.click('#method');cdp.key('ArrowDown');cdp.key('Enter');cdp.wait("document.querySelector('#method').value==='hog_svm' && document.querySelector('#loading').textContent==='Ready'");assert cdp.js("document.querySelector('#processed').dataset.model")== 'hog_svm';assert 'candidate' in cdp.js("document.querySelector('#provenance').textContent");passed('Model/configuration/provenance selection and real HOG/SVM rendering')
        cdp.click('#speed');cdp.key('ArrowDown');cdp.key('Enter');assert cdp.js("document.querySelector('#speed').value")== '2';passed('Playback speed selection')
        cdp.click('#method');cdp.key('ArrowDown');cdp.key('Enter');cdp.wait("document.querySelector('#method').value==='cnn' && document.querySelector('#loading').textContent==='Ready'");passed('Expanded CNN checkpoint and rendering')
        cdp.click('#source');cdp.key('ArrowDown');cdp.key('Enter');cdp.wait("document.querySelector('#source').value==='real' && document.querySelector('#loading').textContent==='Ready'");assert cdp.js("document.querySelector('#processed').dataset.source")== 'real';assert 'starting' in cdp.js("document.querySelector('#provenance').textContent");passed('Earlier real preset resolves starting configuration')
        cdp.click('#source');cdp.key('ArrowDown');cdp.key('Enter');cdp.wait("document.querySelector('#source').value==='synthetic' && document.querySelector('#loading').textContent==='Ready'");passed('Explicit synthetic-trained checkpoint selection')
        cdp.click('#source');cdp.key('ArrowUp');cdp.key('ArrowUp');cdp.key('Enter');cdp.wait("document.querySelector('#source').value==='expanded' && document.querySelector('#loading').textContent==='Ready'")
        cdp.click('#method');cdp.key('ArrowUp');cdp.key('Enter');cdp.wait("document.querySelector('#method').value==='hog_svm' && document.querySelector('#loading').textContent==='Ready'")
        cdp.click('#width');cdp.key('ArrowUp');cdp.key('Enter');cdp.wait("document.querySelector('#width').value==='640' && document.querySelector('#loading').textContent==='Ready'");passed('Preview resolution selection')
        cdp.click('#mode');cdp.key('ArrowDown');cdp.key('Enter');cdp.wait("document.querySelector('#mode').value==='tracks' && document.querySelector('#loading').textContent==='Ready'")
        for _ in range(3):cdp.click('#next');cdp.wait("document.querySelector('#loading').textContent==='Ready'")
        passed('Tracking mode and consecutive frame rendering')
        cdp.click('#seek');cdp.key('End');cdp.click('#cancel');cdp.wait("document.body.dataset.state==='Cancelled'");assert not cdp.js("document.querySelector('#processed').getAttribute('src')");passed('Cancellation clears pending predictions and never reports Ready')
        cdp.click('#seek');cdp.key('Home')
        for _ in range(16):cdp.key('ArrowRight')
        cdp.wait("document.querySelector('#position').textContent.startsWith('Frame 16 /') && document.body.dataset.state==='Ready'");passed('Real tracking seek reaches audited frame 16')
        cdp.click('#save');cdp.wait("document.querySelector('#download').textContent.includes('JPEG')")
        jpeg=urlopen(cdp.js("document.querySelector('#download').href")).read();assert jpeg.startswith(b'\xff\xd8');passed('Save annotated frame and download JPEG')
        displayed=cdp.js("call('frame',selection())");pipeline=cdp.js("document.querySelector('#processed').dataset.pipelineId")
        cdp.click('#export');cdp.wait("document.querySelector('#loading').textContent.includes('Export complete')");passed('Annotated MP4 export')
        href=cdp.js("document.querySelector('#download').href"); data=urlopen(href).read(); path=folder/'browser_export.mp4';path.write_bytes(data)
        import cv2
        cap=cv2.VideoCapture(str(path));count=0
        try:
            while cap.read()[0]:count+=1
        finally:cap.release()
        assert count==30;passed('Export decoded all 30 video frames')
        assert cdp.js("document.querySelector('#download').dataset.pipelineId")==pipeline==displayed['provenance']['pipeline_id']
        prediction=cdp.js("document.querySelector('#download').dataset.predictions");table=pd.read_csv(StringIO(urlopen(base+'/download?file='+prediction+'&token='+server.demo_token).read().decode()))
        expected=pd.DataFrame(displayed['detection_rows']);np.testing.assert_allclose(table.loc[table.frame==16].to_numpy(),expected.to_numpy(),atol=1e-6,rtol=1e-6);passed('Exported frame-16 predictions match displayed model and configuration')
        for view in ['explorer','models','results']:
            cdp.click(f'button[data-view="{view}"]');assert cdp.js(f'!document.querySelector("#{view}").hidden');passed(view+' navigation')
        assert cdp.js("document.querySelectorAll('#stages img').length")==10
        assert cdp.js("Array.from(document.querySelectorAll('#stages img')).every(i=>i.complete&&i.naturalWidth>0)");passed('All ten actual pipeline images loaded')
        for view in ['analyzer','explorer','models','results']:
            cdp.click(f'button[data-view="{view}"]');result=cdp.command('Page.captureScreenshot',{'format':'png','captureBeyondViewport':True});(folder/(view+'.png')).write_bytes(base64.b64decode(result['data']))
        write_json(folder/'browser_results.json',{'status':'passed','checks':checks,'export_decoded_frames':count,'browser':'Headless Edge via CDP; native OS file dialog not opened','limitations':['Browser selected file via DOM.setFileInputFiles; OS chooser appearance not tested','Playback is frame-processing preview, not guaranteed real-time']})
    finally:
        if cdp:
            try:cdp.command('Browser.close')
            except Exception:pass
            cdp.sock.close()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.terminate()
        server.shutdown();server.preview.close();server.server_close();worker.join(5)
        write_json(folder/'checks_recorded.json',checks)
    print(folder,flush=True)

if __name__=='__main__':main()
