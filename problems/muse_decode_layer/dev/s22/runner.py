"""Local checks, quota-aware sends and completion waits for the finite S22 workflow."""
import ctypes,errno,json,os,platform,select,subprocess,sys
from pathlib import Path
import compose
H=compose.H;R=compose.R;ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
sys.path.insert(0,str(R/'tools'));import h200_lab

def run(cmd,log,retry_oom=False):
 attempts=[]
 for attempt in range(3 if retry_oom else 1):
  target=H/(log if attempt==0 else log+f'.retry{attempt}')
  with target.open('w') as out:p=subprocess.run(cmd,cwd=R,env=ENV,stdout=out,stderr=subprocess.STDOUT)
  attempts.append(dict(log=str(target.relative_to(H)),returncode=p.returncode))
  if p.returncode==0:
   if attempt:(H/log).write_text(target.read_text())
   return attempts
  if 'out of memory' not in target.read_text().lower():break
 raise RuntimeError(f'Command failed; inspect {target.relative_to(R)}: {cmd}')

def validate_current(native=False):
 m=json.loads((H/'manifest.json').read_text());candidate=m['shipped_source'];steps={}
 steps['static']=run([sys.executable,'tools/local_eval.py',candidate,'--tier','static','--json',str(H/'generated/static.json')],'static.log')
 for mode in ('normal','forced'):
  cmd=['tools/img','python',str(H/'screen.py'),candidate,'--out',str(H/'screen'/mode)]
  if mode=='forced':cmd+=['--forced']
  steps[mode]=run(cmd,'screen_'+mode+'.log',True)
 if native:steps['native']=run(['tools/img','python',str(H/'native_compile.py')],'native_compile.log')
 steps['packed']=run(['tools/img','python',str(H/'screen_lab.py'),'--forced'],'rehearsal.log',True)
 steps['transport']=run(['tools/img','python','tools/local_eval.py',str(H/'generated/wrapper.py'),'--tier','wrapper','--transport','lab-only','--json',str(H/'generated/transport.json')],'transport.log')
 c=json.loads((H/'contracts.json').read_text());lab=json.loads((H/'generated/lab_forced_validation.json').read_text())
 assert c['source_sha256']==m['sha256']==lab['source_sha256'] and c['passed']
 v=dict(source_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],contracts=len(c['checks']),steps=steps,packed=lab,transport=json.loads((H/'generated/transport.json').read_text()))
 for mode in ('normal','forced'):
  x=json.loads((H/'screen'/mode/'local.json').read_text());assert x['sha256']==m['shipped_sha256'] and all(t['ok'] for t in x['tests']) and len(x['shapes'])==12
  v[mode]=dict(TASK=len(x['tests']),graphs=len(x['shapes']),passed=True)
 (H/'validation.json').write_text(json.dumps(v,indent=2)+'\n');(H/'stages'/f"p{m['phase']}"/'validation.json').write_text(json.dumps(v,indent=2)+'\n')
 return v

def sender_pid(file):
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:args=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if len(args)>2 and args[1].endswith(b'tools/okc_run.py') and file.encode() in args:return int(p.name)
 return None

def receipt_for(sha):
 for line in (R/'runs/ledger.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row.get('sha256')==sha:
   files=list((R/'runs'/row['problem']).glob('*_'+row['job']+'.json'))
   if files:return files[0]
 return None

def send_current():
 m=json.loads((H/'manifest.json').read_text());left=h200_lab.quota();order=['muse_bf16_proj','muse_prefill_layer','muse_moe_w4a16','muse_attention_decode']
 chosen=next((host for host in order if sum(left.get(host,(0,0)))),m['host']);check=not sum(left.get(chosen,(0,0))) or left.get(chosen,(0,0))[1]>0
 if chosen!=m['host']:
  original=(H/'generated/wrapper.py').read_text();payload=m['payload_sha256'];cfg=json.loads((H/'config.json').read_text());cfg['host']=chosen;(H/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
  run([sys.executable,str(H/'build_lab.py')],'build.log');m=json.loads((H/'manifest.json').read_text());assert m['payload_sha256']==payload
  run(['tools/img','python','tools/local_eval.py',str(H/'generated/wrapper.py'),'--tier','wrapper','--transport','lab-only'],'route_transport.log')
 qpath=H/'queue.json'
 if qpath.exists():
  q=json.loads(qpath.read_text())
  if q['wrapper_sha256']==m['wrapper_sha256']:
   p=receipt_for(m['wrapper_sha256'])
   if p:return q
   try:os.kill(q['sender_pid'],0);return q
   except (OSError,KeyError):pass
 note=f"P4 S22 phase{m['phase']} source{m['sha256'][:12]} shipped{m['shipped_sha256'][:12]} vs frozen S21 eacc087b; d16S restored, direct-input hybrids, drops{m['dropped_cells']}, disabled hybrids{m['disabled_hybrids']}; 64 fresh seeds x16 symmetric rounds; S19 per-cell swaps and hybrid arms for every active cell. Contracts, both local TASK/graph screens and exact packed rehearsal passed; preserve all cell ratios, rounds and telemetry."
 decision=f"S22 phase{m['phase']} confirmation vs S21"
 cmd=[sys.executable,'tools/h200_lab.py','send',m['payload'],'--host',chosen,'--lab-only','--phase','import','--import-budget','820','--lane','P4','--decision',decision,'--note',note]
 if check:cmd+=['--check']
 run(cmd,'send.log');pid=sender_pid(m['sent_wrapper']);rp=receipt_for(m['wrapper_sha256']);assert pid or rp,'Sender disappeared without a receipt'
 q=dict(phase=m['phase'],source_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],wrapper_sha256=m['wrapper_sha256'],payload_sha256=m['payload_sha256'],host=chosen,sender_pid=pid,decision=decision,quota_at_send=left,waiting_for_reset=not sum(left.get(chosen,(0,0))),receipt=str(rp.relative_to(R)) if rp else None)
 qpath.write_text(json.dumps(q,indent=2)+'\n');(H/'stages'/f"p{m['phase']}"/'queue.json').write_text(json.dumps(q,indent=2)+'\n');return q

def wait_receipt(q):
 sha=q['wrapper_sha256'];rp=receipt_for(sha)
 if rp:return rp
 pid=q.get('sender_pid');assert platform.machine()=='x86_64'
 # Linux x86_64 __NR_pidfd_open=434, verified in the host's unistd_64.h.
 # Wait for this one sender's completion signal, never a polling cadence.
 if pid:
  libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
  fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(pid),ctypes.c_uint(0))
  if fd<0:
   error=ctypes.get_errno()
   if error!=errno.ESRCH:raise OSError(error,os.strerror(error))
  else:
   try:select.select([fd],[],[])
   finally:os.close(fd)
 rp=receipt_for(sha)
 if rp:return rp
 pending=[json.loads(p.read_text()) for p in (R/'runs/pending').glob('*.json')]
 entry=next((x for x in pending if x.get('sha256')==sha),None)
 if entry:run([sys.executable,'tools/okc_run.py','collect',entry['job']],'collect.log')
 rp=receipt_for(sha)
 if rp is None:raise RuntimeError('S22 sender ended without a receipt; no further action taken')
 return rp

if __name__=='__main__':
 if sys.argv[1:] == ['send']:print(json.dumps(send_current(),indent=2))
 elif sys.argv[1:] == ['validate']:print(json.dumps(validate_current(True),indent=2))
 else:raise SystemExit('runner.py send | validate')
