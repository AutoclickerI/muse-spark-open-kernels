"""Exit immediately on new wrapper receipt completion; blocks on inotify, no timer."""
import ctypes,os,struct,json
from pathlib import Path
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(0)
if fd<0:raise OSError(ctypes.get_errno())
path=Path('runs/muse_prefill_layer')
wd=libc.inotify_add_watch(fd,str(path).encode(),0x8|0x80)
if wd<0:raise OSError(ctypes.get_errno())
seenfile=Path('problems/muse_prefill_layer/dev/watch_seen.json')
seen=set(json.loads(seenfile.read_text())) if seenfile.exists() else set()
def emit(files):
 global seen
 new=sorted(set(files)-seen)
 if not new:return
 for name in new:
  j=json.loads((path/name).read_text());r=j.get('result',{})
  print(json.dumps(dict(receipt=name,job=j['id'],mode=j['mode'],passed=r.get('passed'),speedup=r.get('speedup'),error=r.get('error'))),flush=True)
 seen.update(new);seenfile.write_text(json.dumps(sorted(seen)))
 raise SystemExit(0)
emit(p.name for p in path.glob('*.json'))
print('Waiting for receipt completion; pid',os.getpid(),flush=True)
while True:
 data=os.read(fd,65536);at=0;names=[]
 while at<len(data):
  _,mask,_,size=struct.unpack_from('iIII',data,at);at+=16
  name=data[at:at+size].split(b'\0',1)[0].decode();at+=size
  if name.endswith('.json'):names.append(name)
 emit(names)
