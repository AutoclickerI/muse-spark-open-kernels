"""Render current S22 facts and append only P4's paths to the shared repository."""
import datetime,json,os,re,subprocess
from pathlib import Path
import compose
H=compose.H;R=compose.R

def facts():
 m=json.loads((H/'manifest.json').read_text());q=json.loads((H/'queue.json').read_text()) if (H/'queue.json').exists() else {};s=json.loads((H/'state.json').read_text()) if (H/'state.json').exists() else {}
 r=json.loads((H/'results.json').read_text()) if (H/'results.json').exists() else {};v=json.loads((H/'validation.json').read_text()) if (H/'validation.json').exists() else {}
 return m,q,s,r,v

def description():
 m,q,s,r,v=facts();measured=r.get('phase')==m['phase'] and r.get('measured');score=s.get('scored',{})
 if measured:out=f"native G0 {'passes' if r['lab_gate'] else 'does not pass'}: R {r['score']['R']:.6f} [{r['score']['CI95'][0]:.6f},{r['score']['CI95'][1]:.6f}], prepare {r['prepare']['host_prepare_s']:.2f}s, job{r['job']}"
 elif r and r.get('phase')==m['phase']:out='native evidence unmeasured; job'+r['job']+'; '+r.get('reason','invalid primary/control evidence')
 else:out='native G0/per-cell CIs pending; '+('all hosts quota0 at send, awaiting reset' if q.get('waiting_for_reset') else 'paired lab queued/running')
 out+=f"; phase{m['phase']}, source{m['sha256'][:8]}, shipment{m['shipped_sha256'][:8]}; dropped={m['dropped_cells']}, disabled hybrids={m['disabled_hybrids']}"
 out+=f"; sender{q.get('sender_pid')}, payload{m['payload_sha256'][:16]}"
 if score:out+=f"; scored sender{(score.get('pid') or score.get('job'))} after S21's09:02"
 else:out+=f"; finite follow-through worker{s.get('worker_pid')} selects resolved losers, confirms on new64seeds, scores only after passing G0"
 return out

def update_plan(text):
 m,q,s,r,v=facts();status='- P4 S22 `'+m['sha256'][:8]+'` (shipment `'+m['shipped_sha256'][:8]+'`, [dev/s22](../problems/muse_decode_layer/dev/s22/README.md)): '+description()+'.\n'
 status+='  S21+d16S restore+direct-input v1S/d32S hybrids. Six remaining ported cells have S19 in-graph leave-outs;\n  both hybrids also have prior-port arms. 64freshseeds x16symmetric rounds, all rounds/telemetry retained.\n  CI upper<1 drops a cell/hybrid; each selected composition requires fresh disjoint G0 before scoring.\n  S21 scored sender3225144 at09:02 and S19 sender4046452 at09:01 stay; no new freeze.\n'
 if '- P4 S22 ' in text:text,n=re.subn(r'- P4 S22 .*?(?=^- )',lambda _:status,text,count=1,flags=re.S|re.M);assert n==1
 else:
  assert '- P12 component' in text;text=text.replace('- P12 component',status+'- P12 component',1)
 work="Integration/scoring: board S18 28b5c189 1.8484x; S19 62ea5bf4 G0 1.014463, scored sender4046452 at09:01; accepted S21 109c0135/eacc087b G0 1.002638[1.001195,1.004084],prepare145.86s, scored sender3225144 at09:02 unchanged. S22 "+description()+". S19 d16S restored; direct-input OP v1S and Lee d32S; six S19 per-cell leave-outs plus incremental hybrid arms. Same64x16symmetric telemetry; upperCI<1 selects losses, fresh confirmation required after any selection. [S22](../problems/muse_decode_layer/dev/s22/README.md). No new freeze."
 lines=text.splitlines(True)
 for i,line in enumerate(lines):
  if re.match(r'^\|\s*1\s*\|\s*P4\s*\|',line):
   parts=line.split('|');parts[4]=' '+work+' ';lines[i]='|'.join(parts);break
 else:raise AssertionError('P4 row missing')
 now=datetime.datetime.now().strftime('%Y-%m-%d %H:%M');return re.sub(r'## Status \([^,]+,',f'## Status ({now} KST,',''.join(lines),count=1)

def render():
 m,q,s,r,v=facts();d=set(m['dropped_cells']);disabled=set(m['disabled_hybrids']);rows=[]
 def pick(label):
  if label in ('d1S','d1F','d16S') or label in d:return 'S19'
  if label in ('d16F','d32F'):return 'C3'
  if label=='v32F':return 'Rw4'
  if label in compose.HYBRIDS and label not in disabled:return 'direct OP' if label=='v1S' else 'direct Lee'
  return 'operatorstack' if label=='v1S' else 'Lee'
 text='# S22 integration\n\n'+description()+'.\n\n'
 text+='Base is accepted S21 `eacc087b` / expanded `109c0135`; its scored sender3225144 is unchanged.\nExact12 gate at d1S, native-layout T128 banks, C3 full d16/d32 and v32F Rw4 remain.\n\n| Load | Sliding attention | Full attention |\n|---|---|---|\n'
 for label in ('d1','d16','d32','v1','v16','v32'):text+=f'| {label} | {pick(label+"S")} | {pick(label+"F")} |\n'
 text+='\nThe licensed hybrid patch is `9b043487`, receipt `cde88b9136de4093`; embedded MIT/Apache/NOTICE text and measured sources are preserved. The original d16S port is removed.\n\n'
 text+='Each remaining ported cell has one whole-layer CUDA-graph arm replacing only that cell with exact S19 attention. Both direct-input hybrids also have an arm restoring the previous S21 port, so the d32S nomination whose CI crosses1 is isolated. All active arms get TASK checks on64freshseeds and16symmetric all-round timings; A:A and before/after-block clocks, temperature, power and enqueue traces remain. Per-cell R is time(leave-out)/time(S22): >1 favors retaining the port. `P5B_FEEDBACK.json` supplies the per-cell evidence after the receipt.\n\n'
 text+='The prospectively declared drop rule is an upper95% CI below1. A selected composition uses new disjoint64seeds and new weights/router seeds before scoring. The finite worker stops on a failed/unmeasured final G0 or any execution error; it never silently scores incomplete cell evidence. Primary G0 remains the unchanged held-out six-load lower-bound gate through `tools/promotion.py`; prepare must fit300s. No new freeze precedes official TASK results.\n\n'
 if v.get('source_sha256')==m['sha256']:
  text+=f"Checks: **{v['contracts']} contracts**, normal/forced **{v['normal']['TASK']} TASK + {v['normal']['graphs']} graphs each**; exact packed **{v['packed']['TASK']} TASK + {v['packed']['nearby']} nearby**. Wire **{v['packed']['wire_bytes']} bytes**, full4000-character suffix and exact archive round trip pass.\n\n"
 else:text+='Current build checks are being completed.\n\n'
 text+='Payload SHA256 `'+m['payload_sha256']+'`; wrapper `'+m['wrapper_sha256']+'`.\n\n'
 text+='Current queue/state: [queue.json](queue.json), [state.json](state.json). Worker '+str(s.get('worker_pid'))+' waits on this one sender completion event. Each phase is archived under `stages/pN/`; canonical receipts remain in `runs/`.\n\n'
 for p in sorted((H/'stages').glob('p*/results.json')):
  x=json.loads(p.read_text());text+='Phase '+str(x['phase'])+', job `'+x['job']+'`: '
  if not x.get('measured'):text+='unmeasured; '+x.get('reason','invalid primary/control evidence')+'.\n\n';continue
  text+=f"R **{x['score']['R']:.6f} [{x['score']['CI95'][0]:.6f},{x['score']['CI95'][1]:.6f}]**, prepare **{x['prepare']['host_prepare_s']:.2f}s**.\n\n| Cell/arm | R | 95% CI |\n|---|---:|---|\n"
  for group in ('cells','hybrids'):
   for label,z in x[group].items():
    if z['measured']:text+=f"| {label} ({group}) | {z['R']:.6f} | [{z['CI95'][0]:.6f},{z['CI95'][1]:.6f}] |\n"
    else:text+=f'| {label} ({group}) | unmeasured | |\n'
  text+='\n'
 if s.get('scored'):text+='Official S22 scored sender **'+str(s['scored'].get('pid') or s['scored'].get('job'))+'**, earliest09:03KST10-27, after S21. Official TASK/score remain pending.\n'
 if s.get('error'):text+='Follow-through stopped: '+s['error']+'\n'
 (H/'README.md').write_text(text.rstrip()+'\n');p=R/'docs/PLAN.md';p.write_text(update_plan(p.read_text()))

def commit(message):
 def git(*args,env=None,input=None,check=True):return subprocess.run(['git',*args],cwd=R,env=env,input=input,text=True,capture_output=True,check=check)
 def blob(text):return git('hash-object','-w','--stdin',input=text).stdout.strip()
 m,q,state,r,v=facts();receipts=[x['receipt'] for x in state.get('history',[]) if x.get('receipt')];jobs={x['job'] for x in state.get('history',[]) if x.get('job')}
 paths=['problems/muse_decode_layer/dev/s22','problems/muse_decode_layer/cand_r3_best7_s22_final.py']+sum(([p,str(Path(p).with_suffix('.py'))] for p in receipts),[])
 log_lines=(R/'runs/ledger.jsonl').read_text().splitlines();own=[x for x in log_lines if json.loads(x).get('job') in jobs]
 def ledger(text):
  lines=text.splitlines()
  for line in own:
   job=json.loads(line)['job']
   if not any(json.loads(x).get('job')==job for x in lines):
    positions={x:i for i,x in enumerate(lines)};prior=next((x for x in reversed(log_lines[:log_lines.index(line)]) if x in positions),None);lines.insert(positions[prior]+1 if prior else 0,line)
  return '\n'.join(lines)+'\n'
 env=dict(os.environ,GIT_INDEX_FILE=str(H/'generated/commit-index'))
 assert git('symbolic-ref','HEAD').stdout.strip()=='refs/heads/main'
 for _ in range(3):
  base=git('rev-parse','HEAD').stdout.strip();git('read-tree',base,env=env);git('add','--',*paths,env=env)
  for path,fn in [('docs/PLAN.md',update_plan),('runs/ledger.jsonl',ledger)]:git('update-index','--add','--cacheinfo','100644',blob(fn(git('show',base+':'+path).stdout)),path,env=env)
  names=git('diff','--cached','--name-only',env=env).stdout.splitlines();assert all(n in ('docs/PLAN.md','runs/ledger.jsonl') or any(n==p or n.startswith(p+'/') for p in paths) for n in names)
  git('diff','--cached','--check',env=env);tree=git('write-tree',env=env).stdout.strip();oid=git('commit-tree',tree,'-p',base,input=message+'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n').stdout.strip()
  if git('update-ref','-m','commit: P4 S22','refs/heads/main',oid,base,check=False).returncode==0:break
 else:raise RuntimeError('Concurrent main commits prevented a scoped append')
 updates=[]
 for line in git('ls-tree','-r',oid,'--',*paths).stdout.splitlines():
  meta,path=line.split('\t',1);mode,kind,hash_=meta.split();assert kind=='blob';updates.append(f'{mode} {hash_}\t{path}')
 for path,fn in [('docs/PLAN.md',update_plan),('runs/ledger.jsonl',ledger)]:updates.append('100644 '+blob(fn(git('show',':'+path).stdout))+'\t'+path)
 git('update-index','--index-info',input='\n'.join(updates)+'\n');git('push','origin','main');print('Pushed',oid,flush=True);return oid
