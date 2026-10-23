#!/usr/bin/env python3
"""React to receipt close/rename events; promote verified improvements, no polling.

Only this task's nine sources with matching local correctness+graph evidence are eligible.
Runs go through tools/okc_run.py and retain its usual pending and final receipts.
"""
import argparse,ctypes,hashlib,json,os,selectors,signal,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
dev=Path(__file__).resolve().parent
names=['moe4','moe4_attn','tuned','native','all_decode','glue','moe7','native_glue','hybrid']
manifest={}
for name in names:
 p=dev/f'local_{name}.json'
 if p.exists():
  r=json.loads(p.read_text())
  if len(r.get('tests',[]))==13 and len(r.get('shapes',[]))==12 and all(x['ok'] for x in r['tests']+r['shapes']):
   manifest[r['sha256']]=name
statefile=dev/'promotion_state.json'
state=json.loads(statefile.read_text()) if statefile.exists() else {'promotions':[],'failures':[]}
children={}

def ledger():
 return [r for line in (root/'runs/ledger.jsonl').read_text().splitlines() if (r:=json.loads(line))['problem']=='muse_decode_layer']

def save():
 statefile.write_text(json.dumps(state,indent=2)+'\n')

def update(launch=True):
 rows=ledger()
 for pid,(proc,entry,handle) in list(children.items()):
  if proc.poll() is not None:
   handle.close();entry['exit_code']=proc.returncode
   if proc.returncode:
    state['failures'].append({'check_job':entry['check_job'],'returncode':proc.returncode,'log':entry['log']})
   children.pop(pid);save()
 best=max([1.1428001801346024]+[r.get('speedup') or 0 for r in rows if r['mode']=='submit']+
          [r['speedup'] for r in state['promotions']])
 eligible=[r for r in rows if r['mode']=='check' and r['sha256'] in manifest and (r.get('speedup') or 0)>best]
 if launch and eligible and len(state['promotions'])<8:
  r=max(eligible,key=lambda r:r['speedup'])
  result=json.loads((root/r['source']).with_suffix('.json').read_text()).get('result',{})
  source=root/r['source']
  if not result.get('passed') or hashlib.sha256(source.read_bytes()).hexdigest()!=r['sha256']:
   raise RuntimeError('Receipt or source validation failed for '+r['job'])
  name=manifest[r['sha256']]
  log=dev/f'submit_{name}_{r["job"]}.log'
  handle=log.open('w')
  note=f'Automatic promotion of H200-verified check {r["job"]}: {r["speedup"]:.6f}x, exceeds best completed/pending promotion {best:.6f}x; source matches 13 local tests and 12 fresh graph cases.'
  proc=subprocess.Popen([sys.executable,str(root/'tools/okc_run.py'),'submit',str(source),'--note',note],
                        cwd=root,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
  entry={'check_job':r['job'],'candidate':name,'speedup':r['speedup'],'pid':proc.pid,'log':str(log.relative_to(root))}
  state['promotions'].append(entry);children[proc.pid]=(proc,entry,handle);save()
  print('PROMOTED',json.dumps(entry),flush=True)
 subprocess.run([sys.executable,str(dev/'report_receipts.py')],cwd=root,check=True,stdout=subprocess.DEVNULL)
 checks={r['sha256'] for r in rows if r['mode']=='check' and r['sha256'] in manifest}
 pending=[]
 for path in (root/'runs/pending').glob('*.json'):
  try: pending.append(json.loads(path.read_text()))
  except (FileNotFoundError,json.JSONDecodeError): pass
 unfinished=any(r['problem']=='muse_decode_layer' and r['sha256'] in manifest for r in pending)
 return len(manifest)==9 and len(checks)==9 and not unfinished and not children

ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args()
if args.once:
 update(launch=False);print('Verified local manifest:',len(manifest));sys.exit(0)
libc=ctypes.CDLL(None,use_errno=True)
fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
# IN_CLOSE_WRITE, IN_MOVED_TO, IN_CREATE, IN_DELETE. Watching receipt directories
# avoids waking on our own generated Markdown reports.
for path in [root/'runs/muse_decode_layer',root/'runs/pending']:
 if libc.inotify_add_watch(fd,os.fsencode(path),0x8|0x80|0x200)<0:raise OSError(ctypes.get_errno(),'inotify_add_watch')
rd,wr=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
signal.set_wakeup_fd(wr);signal.signal(signal.SIGCHLD,lambda *_:None)
sel=selectors.DefaultSelector();sel.register(fd,selectors.EVENT_READ);sel.register(rd,selectors.EVENT_READ)
print('Receipt event watcher active; eligible local sources:',len(manifest),flush=True)
while not update():
 for key,_ in sel.select():
  try:os.read(key.fd,1<<20)
  except BlockingIOError:pass
print('All eligible checks and their promotions have final receipts.',flush=True)
