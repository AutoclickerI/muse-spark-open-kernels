"""One diagnostic-completion event, conservative pruning, one paired check.

No polling, sleeps, direct judge API, scoring, background git or agent spawning.
"""
import argparse,ctypes,fcntl,hashlib,json,os,select,subprocess,sys,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;V=H.parent;R=H.parents[4];O=H/'out'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(status,**kw):
 p=H/'queue.json';d=json.loads(p.read_text()) if p.exists() else {};d.update(status=status,**kw)
 tmp=O/('queue-'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def receipts(parent):
 rows=[json.loads(x) for x in (R/'runs/ledger.jsonl').read_text().splitlines()]
 matches=[x for x in rows if x.get('sha256')==parent['sha256']]
 if not matches:return None
 row=matches[-1];paths=list((R/'runs'/row['problem']).glob('*_'+row['job']+'.json'))
 if not paths:
  with (O/'diagnostic_read.log').open('w') as f:subprocess.run([sys.executable,'tools/h200_lab.py','read',row['job']],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  paths=list((R/'runs'/row['problem']).glob('*_'+row['job']+'.json'))
 assert len(paths)==1,'Diagnostic sender has no complete canonical receipt'
 assert sha(paths[0].with_suffix('.py'))==parent['sha256'];return paths[0]
def wait(parent):
 # Check completion only before/after the real sender-completion signal.
 receipt=receipts(parent)
 if receipt:return receipt
 libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
 fd=libc.syscall(ctypes.c_long(434),ctypes.c_int(parent['pid']),ctypes.c_uint(0))
 if fd<0:raise OSError(ctypes.get_errno(),'pidfd_open diagnostic sender')
 try:
  cmd=Path('/proc/'+str(parent['pid'])+'/cmdline').read_bytes()
  assert b'tools/okc_run.py' in cmd and parent['file'].encode() in cmd,'Diagnostic PID identity changed'
  save('waiting on diagnostic completion event',parent=parent,worker_pid=os.getpid());select.select([fd],[],[])
 finally:os.close(fd)
 receipt=receipts(parent);assert receipt,'Diagnostic sender ended without an accepted job/receipt';return receipt
def select_arms(result,manifest):
 if not result.get('measured'):raise RuntimeError('Unmeasured diagnostic; do not infer performance or run dependent kernels blindly')
 points={(x['cell'],x['arm']):x for x in result['points']};keep=list(manifest['arms']);drop={}
 # A different-stage implementation is never closed by another recipe's loss.
 # Only remove inherited complete CHAIN/DIRECT recipes when every changed
 # cell is a resolved loss in the diagnostic (new/control lowerCI >= 1).
 from_cells={'CHAIN':[(c,'D') for c in ('d32S','v1S','v1F','v16S','v16F','v32S')],
             'INC':[(c,'V') for c in ('d1S','d1F')]+[(c,'F') for c in ('d32S','v1S','v1F','v16S','v16F','v32S')]}
 # The diagnostic analyzer labels production cells d/v; enforce that schema.
 for name,cells in from_cells.items():
  rows=[points.get((c,a)) for c,a in cells]
  if all(p and p['new_over_pick']['CI95'][0]>=1 for p in rows):
   keep.remove(name);drop[name]='every changed cell has new/control lower95CI >=1 in diagnostic'
 return dict(keep=keep,dropped=drop,diagnostic_job=result['job'],selection_scope='prune inherited recipes only; STAGE/LANE1/SOFTMAX/OCHAIN remain unmeasured')
def work(expected):
 with (H/'dispatch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  admission=json.loads((H/'admission.json').read_text());assert sha(H/'admission.json')==expected and admission['passed']
  for p,h in admission['sealed_files'].items():assert sha(R/p)==h,'Sealed source changed: '+p
  parent=admission['parent'];receipt=wait(parent)
  with (O/'diagnostic_analysis.log').open('w') as f:subprocess.run([sys.executable,str(V/'analyze.py'),str(receipt)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  result=json.loads((V/'result.json').read_text());selection=select_arms(result,json.loads((H/'manifest.json').read_text()))
  (H/'diagnostic_selection.json').write_text(json.dumps(selection,indent=2)+'\n');save('diagnostic landed; rebuilding selected fixed portfolio',receipt=str(receipt.relative_to(R)),selection=selection)
  keep=O/'keep.json';keep.write_text(json.dumps(selection['keep'])+'\n')
  for p,h in admission['sealed_files'].items():assert sha(R/p)==h,'Sealed source changed while waiting: '+p
  with (O/'selected_build.log').open('w') as f:subprocess.run([sys.executable,str(H/'build_lab.py'),'--keep',str(keep)],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  m=json.loads((H/'lab_manifest.json').read_text())
  for name,s in m['arms'].items():assert s['sha256']==admission['candidate_hashes'][s['logical']]
  # Source bodies and every dispatch were screened before this event. Selection
  # changes only which complete candidates enter this fresh64-seed cohort.
  cmd=[sys.executable,'tools/h200_lab.py','send',str(O/'p5b_night.py'),'--host','muse_attention_decode','--lab-only','--import-budget','840','--check','--lane','P5b','--decision','S22 outcome candidates after in-graph attention diagnostic',
       '--note','P5b after diagnostic'+result['job']+': exact S22 '+m['base_sha256'][:8]+'; screened fixed portfolio '+','.join(selection['keep'])+'; P4 ec754e5 64disjoint seeds x16symmetric cold,all-round means/telemetry/TASK/nearby; whole production graphs only.']
  with (O/'send.log').open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
  wrapper=R/'problems/muse_attention_decode/lab/lab_p5b_night.py';assert sha(wrapper)==m['wrapper_sha256']
  senders=[json.loads(p.read_text()) for p in (R/'runs/waiting').glob('*.json')]
  senders=[s for s in senders if s.get('sha256')==m['wrapper_sha256']]
  ledger=[json.loads(x) for x in (R/'runs/ledger.jsonl').read_text().splitlines()];jobs=[x for x in ledger if x.get('sha256')==m['wrapper_sha256']]
  assert senders or jobs,'No queued sender or accepted job'
  save('one paired check queued; dispatcher complete',paired_senders=senders,paired_jobs=jobs,paired_wrapper_sha256=m['wrapper_sha256'],paired_payload_sha256=m['payload_sha256'])
def start():
 assert not (H/'queue.json').exists(),'Inspect existing one-shot queue before starting another'
 expected=sha(H/'admission.json');O.mkdir(exist_ok=True)
 with (O/'dispatch.log').open('a') as f:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--admission',expected],cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 save('dependency dispatcher started',worker_pid=p.pid,admission_sha256=expected,one_paired_check_only=True)
 print('P5b paired dependency worker',p.pid)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('start','run'));ap.add_argument('--admission');a=ap.parse_args()
 if a.action=='start':start()
 else:
  try:work(a.admission)
  except Exception as e:save('needs attention; no extra send',error=type(e).__name__+':'+str(e),traceback=traceback.format_exc());raise
