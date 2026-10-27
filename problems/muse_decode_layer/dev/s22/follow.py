"""Finite, sealed S22 follow-through: await, select, freshly confirm, then score.

No periodic polling/advice. The worker waits on the exact sender's pidfd, stops
if its manifest/config changes, and makes at most eight monotone pruning steps.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,traceback
from pathlib import Path
import compose
H=compose.H;R=compose.R
runner=compose.load(H/'runner.py','s22_follow_runner')
report=compose.load(H/'report.py','s22_follow_report')
finish=compose.load(H/'finish.py','s22_follow_finish')

def decide(result,config):
 if not result.get('measured'):raise RuntimeError('Primary paired evidence is unmeasured')
 for group in ('cells','hybrids'):
  if any(not v['measured'] for v in result[group].values()):raise RuntimeError('Missing requested per-cell/hybrid evidence')
 # A fresh confirmation of a selection that passes G0 is scored; losses it resolves go to the next integration batch.
 if config['phase']>0 and result['lab_gate']:return dict(action='score')
 dropped=set(config['dropped_cells']);disabled=set(config['disabled_hybrids'])
 dropped.update(n for n,v in result['cells'].items() if v['CI95'][1]<1)
 disabled.update(n for n,v in result['hybrids'].items() if v['CI95'][1]<1)
 disabled-=dropped
 if dropped!=set(config['dropped_cells']) or disabled!=set(config['disabled_hybrids']):
  return dict(action='confirm',config=dict(phase=config['phase']+1,host=config['host'],dropped_cells=sorted(dropped),disabled_hybrids=sorted(disabled)))
 return dict(action='score' if result['lab_gate'] else 'stop')

def save(state):
 p=H/'state.json';tmp=H/'generated/state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)

def score_current(m,r):
 path=R/m['shipped_source'];assert hashlib.sha256(path.read_bytes()).hexdigest()==m['shipped_sha256']
 assert r['lab_gate'] and r['source_sha256']==m['sha256']
 assert all(v['measured'] for group in ('cells','hybrids') for v in r[group].values())
 loads=', '.join(f"{k} {v['R']:.4f}" for k,v in r['score']['loads'].items())
 note=f"P4 S22 phase{m['phase']} source{m['sha256'][:12]} shipped{m['shipped_sha256'][:12]}: S21 + S19 d16S + direct-input v1S/d32S hybrids, drops{m['dropped_cells']}, disabled hybrids{m['disabled_hybrids']}. Native fresh-seed G0 over frozen S21 eacc087b ({r['job']}, 64 seeds x16 symmetric rounds): R{r['score']['R']:.6f} CI{[round(x,6) for x in r['score']['CI95']]}; {loads}; in-lab TASK {r['checks']['task_checks']}+{r['checks']['nearby_task_checks']} nearby pass; prepare{r['prepare']['host_prepare_s']:.2f}s."
 cmd=[sys.executable,'tools/okc_run.py','send',m['shipped_source'],'--scored','--lane','P4','--decision',f"S22 phase{m['phase']} official score",'--note',note]
 runner.run(cmd,'score_send.log');pid=runner.sender_pid(m['shipped_source'])
 if pid:return dict(pid=pid,sha256=m['shipped_sha256'])
 rp=runner.receipt_for(m['shipped_sha256'])
 if rp:return dict(pid=None,sha256=m['shipped_sha256'])
 raise RuntimeError('Scored send left neither a live sender nor a receipt')

def freeze(m,r,scored):
 """Wait for the official scored receipt; freeze only when the judge's checks pass."""
 rp=runner.wait_receipt(dict(wrapper_sha256=m['shipped_sha256'],sender_pid=scored.get('pid')));x=json.loads(rp.read_text());res=x['result']
 row=next(json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().splitlines() if json.loads(l).get('job')==x['id'])
 ok=row['mode']=='submit' and res.get('passed') is True and res.get('checks',{}).get('ok') is True and res.get('speedup')
 setup=next((y for y in res.get('records',[]) if y.get('event')=='setup' and y.get('prepare_s')),{})
 f=dict(name='S22',phase=m['phase'],expanded_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],file=m['shipped_source'],dropped_cells=m['dropped_cells'],disabled_hybrids=m['disabled_hybrids'],
  scored_job=x['id'],scored_speedup=res.get('speedup'),scored_interval=res.get('speedup_ci'),scored_receipt=str(rp.relative_to(R)),official_prepare_s=setup.get('prepare_s'),
  official_TASK_run=f"{x['id']} ({'checks ok' if ok else 'FAILED: passed='+str(res.get('passed'))+' checks='+json.dumps(res.get('checks'))})",
  confirmation_job=r['job'],confirmation_host=m['host']+' import-time lab',R=r['score']['R'],CI95=r['score']['CI95'],loads={k:v['R'] for k,v in r['score']['loads'].items()},over='S21 109c0135/eacc087b',
  seeds=r['score']['n'],rounds=r['measurement_protocol']['rounds'],TASK_checks=r['checks']['task_checks'],nearby_TASK_checks=r['checks']['nearby_task_checks'],prepare_s=r['prepare']['host_prepare_s'],
  licenses='Lee v56 (Apache-2.0) and operatorstack (MIT) notices shipped, R-20261026-1')
 if not ok:raise RuntimeError('Official scored run did not pass: '+f['official_TASK_run'])
 (H/'frozen.json').write_text(json.dumps(f,indent=2)+'\n');return f

def work(expected):
 with (H/'followthrough.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  _work_locked(expected)

def _work_locked(expected):
 signal.signal(signal.SIGTERM,lambda *_:sys.exit(143))
 state=json.loads((H/'state.json').read_text()) if (H/'state.json').exists() else dict(history=[])
 state.update(worker_pid=os.getpid(),status='awaiting_native_lab');save(state)
 try:
  for _ in range(9):
   m=json.loads((H/'manifest.json').read_text());cfg=json.loads((H/'config.json').read_text());q=json.loads((H/'queue.json').read_text())
   assert m['wrapper_sha256']==expected==q['wrapper_sha256'],'Manifest changed; stop stale follow-through'
   assert all(cfg[k]==m[k] for k in ('phase','host','dropped_cells','disabled_hybrids')),'Config changed; stop stale follow-through'
   rp=runner.wait_receipt(q)
   assert json.loads((H/'manifest.json').read_text())['wrapper_sha256']==expected,'Manifest changed while lab ran'
   result=report.analyze(rp,H/'stages'/f"p{m['phase']}"/'manifest.json');choice=decide(result,cfg)
   history=state.setdefault('history',[]);entry=dict(phase=m['phase'],job=result['job'],receipt=str(rp.relative_to(R)),source_sha256=m['sha256'],R=result.get('score',{}).get('R'),CI95=result.get('score',{}).get('CI95'),decision=choice['action'])
   history[:]=[x for x in history if x['phase']!=m['phase']];history.append(entry)
   q.update(receipt=str(rp.relative_to(R)),lab_complete=True,job=result['job']);(H/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
   if choice['action']=='score':
    state.update(status='scored_queued',scored=score_current(m,result));save(state);finish.render();finish.commit(f"P4: S22 phase{m['phase']} passes fresh G0 over S21; queue its scored run")
    f=freeze(m,result,state['scored']);state.update(status='frozen',frozen=f);save(state);finish.render();finish.commit(f"P4: freeze S22 phase{m['phase']} at {f['scored_speedup']:.4f}x (official TASK pass)");return
   if choice['action']=='stop':
    state['status']='G0_failed';save(state);finish.render();finish.commit(f"P4: retain measured S22 phase{m['phase']} negative G0 and per-cell evidence");return
   assert choice['config']['phase']<=8,'More than eight monotone selection steps'
   state.update(status='selection_requires_fresh_confirmation',selected_config=choice['config']);save(state);finish.render();finish.commit(f"P4: measure S22 phase{m['phase']} and select resolved per-cell losses for fresh confirmation")
   (H/'config.json').write_text(json.dumps(choice['config'],indent=2)+'\n')
   runner.run([sys.executable,str(H/'contracts.py')],'contracts.log')
   runner.run([sys.executable,str(H/'build_lab.py')],'build.log')
   runner.validate_current(native=False)
   q=runner.send_current();expected=q['wrapper_sha256'];state.update(status='awaiting_fresh_confirmation',phase=choice['config']['phase']);save(state);finish.render();finish.commit(f"P4: screen selected S22 phase{choice['config']['phase']} and queue disjoint G0 confirmation")
  raise RuntimeError('Finite S22 phase bound exhausted')
 except Exception as exc:
  state.update(status='needs_attention',error=type(exc).__name__+': '+str(exc));save(state)
  # Do not commit another owner's edits if the sealed manifest was replaced.
  if json.loads((H/'manifest.json').read_text())['wrapper_sha256']==expected:finish.render()
  traceback.print_exc();raise

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('start','run'));ap.add_argument('--expected-wrapper');a=ap.parse_args()
 if a.action=='start':
  q=json.loads((H/'queue.json').read_text());state_path=H/'state.json'
  if state_path.exists():
   old=json.loads(state_path.read_text());pid=old.get('worker_pid')
   if pid:
    try:
     args=(Path('/proc')/str(pid)/'cmdline').read_bytes()
     if str(Path(__file__).resolve()).encode() in args:raise SystemExit(f'S22 worker already active: {pid}')
    except FileNotFoundError:pass
  with (H/'followthrough.log').open('a') as log:
   p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run','--expected-wrapper',q['wrapper_sha256']],cwd=R,env=runner.ENV,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (H/'worker.json').write_text(json.dumps(dict(pid=p.pid,wrapper_sha256=q['wrapper_sha256'],finite_max_phases=9,wait='sender completion pidfd; no polling timer'),indent=2)+'\n');print('S22 follow-through worker',p.pid)
 else:work(a.expected_wrapper)
