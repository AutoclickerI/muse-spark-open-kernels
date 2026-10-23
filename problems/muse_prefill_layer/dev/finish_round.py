"""Event-driven completion of the authorized prefill round.

Development-only: collect processes handle receipts. This process promotes and
scores new checked bests, then locally screens/checks one measured integration.
No code from this file is embedded in a kernel submission.
"""
import argparse,ctypes,datetime,fcntl,hashlib,json,os,select,subprocess,sys,traceback
from collections import Counter
from pathlib import Path
import build_measured_combo as combo
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1];DEV=P/'dev'
STATE=DEV/'finish_round_state.json'
DAY='20261023'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
def load():return json.loads(STATE.read_text()) if STATE.exists() else {'scoring':{},'integration':None}
def save(s):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(STATE)
def log(s):print(now(),s,flush=True)
def rows():
 rs=[]
 for l in (ROOT/'runs/ledger.jsonl').read_text().splitlines():
  try:rs.append(json.loads(l))
  except json.JSONDecodeError:pass
 for f in (ROOT/'runs/pending').glob('*.json'):
  try:rs.append(json.loads(f.read_text()))
  except (OSError,json.JSONDecodeError):pass
 return [r for r in rs if r.get('problem')=='muse_prefill_layer']
def launch(mode,path,note):
 f=open(DEV/f'auto_{mode}_{path.stem}.log','a')
 proc=subprocess.Popen([sys.executable,str(ROOT/'tools/okc_run.py'),mode,str(path),'--note',note],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 f.close();return proc.pid

def refresh_status(state,missing):
 today=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
 current_sha=hashlib.sha256((P/'submission.py').read_bytes()).hexdigest()
 current=[]
 for f in (ROOT/'runs/muse_prefill_layer').glob('*_check_*.json'):
  source=f.with_suffix('.py')
  if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest()==current_sha:
   d=json.loads(f.read_text());current.append((d['result'].get('speedup',0),d['id']))
 current_text=f'check{max(current)[1]}, {max(current)[0]:.9f}x' if current else 'no matching completed check receipt'
 allrows={r['job']:r for r in rows() if r.get('utc','').startswith(today)}
 counts=Counter(r['mode'] for r in allrows.values())
 waiting=[]
 for f in (ROOT/'runs/waiting').glob('*.json'):
  try:r=json.loads(f.read_text())
  except (OSError,json.JSONDecodeError):continue
  if r.get('problem')=='muse_prefill_layer' and r.get('priority',2)>=0:waiting.append(r)
 pending=[f"{r['mode']} {j} ({r.get('file','')})" for j,r in allrows.items() if r.get('status')!='done']
 integration=state.get('integration')
 stage=integration['status'] if integration else 'awaiting component receipts'
 board=DEV/'board_resumed.json';board_text='not queried'
 if board.exists():
  b=json.loads(board.read_text()).get('you',{});board_text=f"rank{b.get('rank')}, {b.get('speedup')}x"
 text=f'''# Active prefill round

Current snapshot: {now()}. This file is rewritten by the receipt handler; experiment history is in runs receipts and RESULTS.md.

Owner instructions: the two-failed-ideas stop condition was revoked. Do useful development while judge jobs run. Edit only this problem; the original no-commit hold remains. Keep accepted candidate sources immutable. Use tools/okc_run.py for every remote check/submit, with a note. Count accepted and waiting checks before sending; the external limit is20/day, resetting00:00UTC/09:00KST.

Current submission.py: {current_text}. SHA256 {current_sha}. Last verified board: {board_text}. Check performance and board performance are distinct; pending scored runs do not establish a new board result.

Accepted this UTC day: {counts['check']}/20 checks, {counts['submit']}/10 scored. Prefill check admission requests: {len(waiting)}. Accepted plus waiting checks: {counts['check']+len(waiting)}/20. Final integration status: {stage}. Missing component receipts: {missing}.

Pending accepted jobs:
'''+('\n'.join('- '+x for x in pending) or '- none')+f'''

Background result handler: dev/finish_round.py, PID{os.getpid()}, logdev/finish_round.log. It watches actual receipt events, promotes exact passing checked sources, scores through the wrapper within quota, excludes failed scored reruns and rolls back if necessary. Each promoted source is frozen. A single-instance lock prevents duplicate handlers. It exits when this round's checks and scored jobs finish. Collectors and admission wrappers are independent processes; do not relaunch accepted jobs.

Final integration builder: dev/build_measured_combo.py. It selects experts by token-size bucket, attention by token/window buckets, and native dense from correctedv13 measurements. It statically prefixes private expert definitions; submitted code imports no other candidate and does not execute generated Python or start processes. The final check is sent only after official local tests, fresh nearby graphs, forced first-turn local tests and Hopper Python-glue stand-ins pass. The assembled research state may retain multiple weight layouts; share a layout or choose one family before full-model serving.

v14–v17 all passed5official local tests. v15 shared-A gate/down emulation passes; v17 covers every finite signed E4M3 scale and all FP4 codes with0.159% local GEMM relativeL2. v15/v17 sm90a builds have no spills. Hopper TMA flags forced onto unsupported sm120 failed; the actual guardedv14 passes. Actual FA3/WGMMA/TMA performance is verified only by H200 receipts. Combined and forced-first-turn rehearsals pass local layer/graph/glue checks; these establish no H200 speedup.

Sources and current evidence: RESULTS.md, PORTING.md, dev/results_summary.json/txt. Builders refuse source overwrite. Upstream sync in own-directory staging found official prefill files and llms unchanged. No serving edits or commits were made.

Next hypotheses when check capacity is available: fast decode plus256-token tiles if both measured components win; a direct packed-bank dequantizer for a memory-efficient custom/DG serving hybrid; lower-register/shallower-pipeline BN64 occupancy tuning if turn1 remains expensive. None is a measured H200 gain yet.
'''
 (DEV/'LIVE_STATUS.md').write_text(text)

def audit(state,dry):
 rs=combo.receipts();eligible=[]
 bad=combo.failed_score_hashes()
 # Include the eventual integrated candidate, whose ID is assigned on admission.
 for f in (ROOT/'runs/muse_prefill_layer').glob('*_check_*.json'):
  try:d=json.loads(f.read_text())
  except (OSError,json.JSONDecodeError):continue
  r=d.get('result',{})
  if combo.passed(d):
   source=f.with_suffix('.py')
   sha=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
   if sha in state['allowed_sha'] and sha not in bad:
    eligible.append((r['speedup'],d,source))
 allrows=rows()
 if eligible:
  speed,d,source=max(eligible,key=lambda x:x[0]);blob=source.read_bytes();sha=hashlib.sha256(blob).hexdigest()
  if (P/'submission.py').read_bytes()!=blob:
   log(f'Promote checked best {d["id"]}, {speed:.9f}x, sha256={sha}')
   if not dry:
    (P/'submission.py').write_bytes(blob)
    with (DEV/'promotions.jsonl').open('a') as f:f.write(json.dumps(dict(utc=now(),job=d['id'],speedup=speed,sha256=sha,source=str(source.relative_to(ROOT))))+'\n')
  already=any(r.get('mode')=='submit' and r.get('sha256')==sha for r in allrows)
  if already or sha in state['scoring']:state.pop('needs_scoring',None)
  if not already and sha not in state['scoring']:
   day=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
   accepted=[r for r in allrows if r.get('mode')=='submit' and r.get('utc','').startswith(day)]
   admitted_hash={r.get('sha256') for r in allrows if r.get('mode')=='submit'}
   reserved=[v for h,v in state['scoring'].items() if h not in admitted_hash]
   if len(accepted)+len(reserved)<10:
    log(f'Score checked best {d["id"]}, {speed:.9f}x')
    if not dry:
     frozen=P/f'promoted_{d["id"]}.py'
     if frozen.exists() and frozen.read_bytes()!=blob:raise RuntimeError('immutable promoted source mismatch')
     frozen.write_bytes(blob)
     pid=launch('submit',frozen,f'Automatic authorized promotion: H200 check {d["id"]} passed {speed:.9f}x. Best passing checked source from this round; scored failures excluded. Exact checked source; no kernel edits after check.')
     state['scoring'][sha]=dict(utc=now(),check=d['id'],file=str(frozen.relative_to(ROOT)),pid=pid)
     state.pop('needs_scoring',None)
     save(state)
   else:
    log('Scored quota reserved/used; promoted source awaits a scored slot.')
    if not dry:state['needs_scoring']=sha;save(state)
 missing=sorted(set(combo.REQUIRED)-rs.keys())
 if not missing and state['integration'] is None:
  pl=combo.plan(rs)
  changed=any(v!='paged' for v in pl['attention'].values()) or any(v!='cublas' for v in pl['dense'].values()) or any(v!=pl['base'] for v in pl['experts'].values())
  log('All component receipts complete. Final measured plan: '+json.dumps(pl))
  if not changed:
   if not dry:state['integration']={'status':'redundant','utc':now(),'plan':pl};save(state)
  elif not dry:
   # Reserve the twentieth check only after every component has a real receipt.
   day=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
   used={r['job'] for r in allrows if r.get('mode')=='check' and r.get('utc','').startswith(day)}
   waiting=[]
   for f in (ROOT/'runs/waiting').glob('*.json'):
    try:r=json.loads(f.read_text())
    except (OSError,json.JSONDecodeError):continue
    if r.get('problem')=='muse_prefill_layer' and r.get('priority',2)>=0:waiting.append(r)
   if len(used)+len(waiting)>=20:
    state['integration']={'status':'quota_exhausted','utc':now(),'plan':pl};save(state)
    log('Final integration retained as plan; the current daily check capacity is exhausted.')
   else:
    output=P/'cand_v18.py';combo.build(pl,output)
    state['allowed_sha'].append(hashlib.sha256(output.read_bytes()).hexdigest())
    (DEV/'combo_plan.json').write_text(json.dumps(pl,indent=2))
    state['integration']={'status':'screening','utc':now(),'plan':pl,'file':str(output.relative_to(ROOT))};save(state)
    local=DEV/'local_v18.log'
    with local.open('w') as f:
     cmd=[str(ROOT/'tools/img'),'python',str(ROOT/'tools/local_eval.py'),str(output),'--tests-only','--json',str(DEV/'local_v18.json')]
     env=dict(os.environ,CUDA_VISIBLE_DEVICES='0')
     rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc==0:
     with (DEV/'verify_graph_v18.log').open('w') as f:
      rc=subprocess.run([str(ROOT/'tools/img'),'python',str(DEV/'verify_graph.py'),str(output)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc==0:
     # Official local tests stop at4096 tokens. Force the actual first-turn
     # selection on those tests so its namespaced Python/Triton glue is reached.
     import re
     forced=DEV/'cand_v18_first_local.py'
     code,n=re.subn(r'(?m)^def _load_bucket\(t\):\n    return[^\n]+',"def _load_bucket(t):\n    return 'first turn'",output.read_text())
     if n!=1:raise RuntimeError('Unable to force first-turn dispatch for local screening')
     forced.write_text(code)
     with (DEV/'local_v18_first.log').open('w') as f:
      rc=subprocess.run([str(ROOT/'tools/img'),'python',str(ROOT/'tools/local_eval.py'),str(forced),'--tests-only','--json',str(DEV/'local_v18_first.json')],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc==0:
     with (DEV/'verify_hopper_v18.log').open('w') as f:
      rc=subprocess.run([str(ROOT/'tools/img'),'python',str(DEV/'verify_hopper_glue.py'),str(output)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    if rc:
     state['integration']['status']='local_failed';save(state);log('Final integration failed local screen; no remote check sent.')
    else:
     pid=launch('check',output,'v18 measured integration: expert family from best completed H200 whole-layer expert trial; per-load/layer attention selected from v2/v4/v9 profiles; native dense only where corrected v13/control comparison supports it. All local official and nearby fresh graph tests pass. Plan in dev/combo_plan.json.')
     state['integration'].update(status='check_launched',pid=pid);save(state)
     log(f'Final integration passed local screen; wrapper launched pid={pid}')
 if not dry:
  completed_scores=[]
  for f in (ROOT/'runs/muse_prefill_layer').glob('*_submit_*.json'):
   try:d=json.loads(f.read_text());completed_scores.append(d['id'])
   except (OSError,json.JSONDecodeError):pass
  if sorted(completed_scores)!=state.get('board_jobs'):
   board=subprocess.run([str(Path.home()/'.okc/bin/okc'),'leaderboard','muse_prefill_layer','--json'],cwd=ROOT,capture_output=True,text=True)
   if board.returncode==0:
    (DEV/'board_resumed.json').write_text(board.stdout);state['board_jobs']=sorted(completed_scores);save(state)
  subprocess.run([sys.executable,str(DEV/'write_report.py')],cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
  refresh_status(state,missing)
 return missing

def completed(state):
 if state.get('needs_scoring'):return False
 if not state.get('integration'):return False
 status=state['integration']['status']
 if status=='screening':return False
 rs=rows()
 if status=='check_launched':
  path=ROOT/state['integration']['file']
  sha=hashlib.sha256(path.read_bytes()).hexdigest()
  if not any(r.get('mode')=='check' and r.get('sha256')==sha and r.get('status')=='done' for r in rs):return False
 # Accepted scored jobs and not-yet-admitted scored reservations must all land.
 if any(r.get('mode')=='submit' and r.get('utc','').startswith(DAY) and r.get('status')!='done' for r in rs):return False
 done={r.get('sha256') for r in rs if r.get('mode')=='submit' and r.get('status')=='done'}
 return all(h in done for h in state['scoring'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');ap.add_argument('--dry-run',action='store_true');a=ap.parse_args()
 lock=(DEV/'finish_round.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
 state=load()
 if 'allowed_sha' not in state:
  state['allowed_sha']=[hashlib.sha256((P/f'cand_v{n}.py').read_bytes()).hexdigest() for n in range(1,18)]
  if not a.dry_run:save(state)
 # Install watches before the first scan; no completion can be missed between them.
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_CLOEXEC)
 if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
 mask=0x8|0x80 # IN_CLOSE_WRITE | IN_MOVED_TO
 for path in (ROOT/'runs/muse_prefill_layer',ROOT/'runs/pending'):
  if libc.inotify_add_watch(fd,os.fsencode(path),mask)<0:raise OSError(ctypes.get_errno(),str(path))
 missing=audit(state,a.dry_run);log('Pending component receipts: '+str(missing))
 if a.once:return
 if not missing and completed(state):log('Authorized round completed; exiting.');return
 while True:
  timeout=None
  if state.get('needs_scoring'):
   utc=datetime.datetime.now(datetime.timezone.utc)
   reset=(utc+datetime.timedelta(days=1)).replace(hour=0,minute=0,second=0,microsecond=0)
   timeout=(reset-utc).total_seconds() # Actual external quota-reset event, no polling cadence.
  ready,_,_=select.select([fd],[],[],timeout)
  if ready:os.read(fd,65536)
  try:
   missing=audit(state,a.dry_run)
   if not missing and completed(state):log('Authorized round completed; exiting.');return
  except Exception:traceback.print_exc();log('Audit failed; preserved state and waiting for next receipt event.')
if __name__=='__main__':main()
