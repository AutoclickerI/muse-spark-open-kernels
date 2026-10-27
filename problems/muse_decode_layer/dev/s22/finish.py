"""Render current S22 facts and append only P4's paths to the shared repository.

PLAN is orchestrator-owned prose: refresh only P4's own S22 Status bullet and the
P4 Ranked-work cell when they exist, and never assert on any other wording.
"""
import json,os,re,subprocess,textwrap
from pathlib import Path
import compose
H=compose.H;R=compose.R
BASE='S21 `eacc087b` / expanded `109c0135`, frozen at official 1.8781x (`ec8cc95c`)'

def facts():
 m=json.loads((H/'manifest.json').read_text());q=json.loads((H/'queue.json').read_text()) if (H/'queue.json').exists() else {};s=json.loads((H/'state.json').read_text()) if (H/'state.json').exists() else {}
 r=json.loads((H/'results.json').read_text()) if (H/'results.json').exists() else {};v=json.loads((H/'validation.json').read_text()) if (H/'validation.json').exists() else {}
 return m,q,s,r,v

def ci(x):return f"R {x['R']:.6f} [{x['CI95'][0]:.6f},{x['CI95'][1]:.6f}]"

def description():
 m,q,s,r,v=facts();out=[]
 for x in s.get('history',[]):
  if x['phase']<m['phase'] and x.get('R') is not None:out.append(f"phase{x['phase']} {ci(x)} (`{x['job'][:8]}`)")
 if m['phase']>0:out.append(f"selected drop={m['dropped_cells']}, disabled hybrids={m['disabled_hybrids']}")
 if r.get('phase')==m['phase'] and r.get('measured'):out.append(f"phase{m['phase']} native G0 {'passes' if r['lab_gate'] else 'fails'}: {ci(r['score'])}, prepare {r['prepare']['host_prepare_s']:.2f}s (`{r['job'][:8]}`)")
 elif r.get('phase')==m['phase']:out.append(f"phase{m['phase']} unmeasured (`{r['job'][:8]}`): {r.get('reason','invalid primary/control evidence')}")
 else:out.append(f"phase{m['phase']} fresh-seed G0 vs S21 pending (sender {q.get('sender_pid')})")
 f=s.get('frozen');sc=s.get('scored')
 if f:out.append(f"official {f['scored_speedup']:.4f}x (`{f['scored_job'][:8]}`), TASK pass, frozen")
 elif sc:out.append(f"official scored run {sc.get('job') or 'sender '+str(sc.get('pid'))} pending")
 return '; '.join(out)

def update_plan(text):
 m,q,s,r,v=facts();d=description()
 bullet=textwrap.fill(f"- S22 `{m['sha256'][:8]}` (shipment `{m['shipped_sha256'][:8]}`, [dev/s22](../problems/muse_decode_layer/dev/s22/README.md)): S21 + S19 d16S + direct-input v1S/d32S hybrids; {d}.",width=116,subsequent_indent='  ',break_on_hyphens=False,break_long_words=False)+'\n'
 work=f"Integration/scoring: {BASE.replace('`','')}. S22 {m['sha256'][:8]}/{m['shipped_sha256'][:8]}: {d}. Per-cell S19 leave-outs and hybrid arms on 64 fresh seeds x16 symmetric rounds; upper CI<1 selects a loss, each selection needs a fresh disjoint G0 before scoring. [S22](../problems/muse_decode_layer/dev/s22/README.md)."
 lines=text.splitlines(True);out=[];i=0;bullet_done=False
 while i<len(lines):
  line=lines[i]
  if not bullet_done and re.match(r'- (P4 )?S22 `',line):
   out.append(bullet);bullet_done=True;i+=1
   while i<len(lines) and lines[i].startswith('  ') and lines[i].strip():i+=1
   continue
  parts=line.split('|')
  if len(parts)>=9 and re.fullmatch(r'\s*P4\s*',parts[2]):parts[4]=' '+work+' ';line='|'.join(parts)
  out.append(line);i+=1
 return ''.join(out)

def render():
 m,q,s,r,v=facts();d=set(m['dropped_cells']);disabled=set(m['disabled_hybrids'])
 def pick(label):
  if label in ('d1S','d1F','d16S') or label in d:return 'S19'
  if label in ('d16F','d32F'):return 'C3'
  if label=='v32F':return 'Rw4'
  if label in compose.HYBRIDS and label not in disabled:return 'direct OP' if label=='v1S' else 'direct Lee'
  return 'operatorstack' if label=='v1S' else 'Lee'
 text='# S22 integration\n\n'+description()+'.\n\n'
 text+=f"Base is {BASE}. Exact12 gate at d1S, native-layout T128 banks, C3 full d16/d32 and v32F Rw4 remain.\nCurrent shipment: `{m['shipped_source']}` (phase {m['phase']}).\n\n| Load | Sliding attention | Full attention |\n|---|---|---|\n"
 for label in ('d1','d16','d32','v1','v16','v32'):text+=f'| {label} | {pick(label+"S")} | {pick(label+"F")} |\n'
 text+='\nThe licensed hybrid patch is `9b043487`, receipt `cde88b9136de4093`; embedded MIT/Apache/NOTICE text and measured sources are preserved. The S21 d16S port is removed (S19 at d16S).\n\n'
 text+='Each remaining ported cell has one whole-layer CUDA-graph arm replacing only that cell with exact S19 attention; each active direct-input hybrid also has an arm restoring the previous S21 port. All arms get TASK checks on 64 fresh seeds and 16 symmetric all-round timings, with A:A and before/after-block clocks, temperature, power and enqueue traces. Per-cell R is time(leave-out)/time(S22): >1 favors keeping the port. `P5B_FEEDBACK.json` holds the per-cell evidence.\n\n'
 text+='Rule: a cell or hybrid whose upper 95% CI is below 1 is selected for removal, and a selected composition is confirmed on new disjoint seeds and new weight/router seeds before scoring. A confirmation whose lower bound is above 1 (GOAL G0 through `tools/promotion.py`, in-lab TASK and nearby checks complete, prepare <= 300 s) is scored; per-cell losses it resolves go to the next integration batch. `TASK_pass` in `results.json` is false on lab-only hosts by construction (their receipt fails setup after the import-time lab); the official TASK result comes from the scored run.\n\n'
 if v.get('source_sha256')==m['sha256']:
  text+=f"Checks: **{v['contracts']} contracts**, normal/forced **{v['normal']['TASK']} TASK + {v['normal']['graphs']} graphs each**; exact packed **{v['packed']['TASK']} TASK + {v['packed']['nearby']} nearby**. Wire **{v['packed']['wire_bytes']} bytes**, full4000-character suffix and exact archive round trip pass.\n\n"
 else:text+='Current build checks are being completed.\n\n'
 text+='Payload SHA256 `'+m['payload_sha256']+'`; wrapper `'+m['wrapper_sha256']+'`. Queue/state: [queue.json](queue.json), [state.json](state.json); each phase is archived under `stages/pN/`; canonical receipts are in `runs/`.\n\n'
 for p in sorted((H/'stages').glob('p*/results.json')):
  x=json.loads(p.read_text());text+='Phase '+str(x['phase'])+', job `'+x['job']+'`: '
  if not x.get('measured'):text+='unmeasured; '+x.get('reason','invalid primary/control evidence')+'.\n\n';continue
  text+=f"{ci(x['score'])}, prepare **{x['prepare']['host_prepare_s']:.2f}s**; per load "+', '.join(f"{k} {z['R']:.4f}" for k,z in x['score']['loads'].items())+".\n\n| Cell/arm | R | 95% CI |\n|---|---:|---|\n"
  for group in ('cells','hybrids'):
   for label,z in x[group].items():
    if z['measured']:text+=f"| {label} ({group}) | {z['R']:.6f} | [{z['CI95'][0]:.6f},{z['CI95'][1]:.6f}] |\n"
    else:text+=f'| {label} ({group}) | unmeasured | |\n'
  text+='\n'
 f=s.get('frozen')
 if f:text+=f"Frozen: phase {f['phase']} `{f['file']}`, official **{f['scored_speedup']:.4f}x** (job `{f['scored_job']}`, checks ok); see [frozen.json](frozen.json).\n"
 elif s.get('scored'):text+='Official scored run '+str(s['scored'].get('job') or 'sender '+str(s['scored'].get('pid')))+' is pending; no freeze before its TASK result.\n'
 if s.get('error'):text+='Follow-through stopped: '+s['error']+'\n'
 (H/'README.md').write_text(text.rstrip()+'\n');p=R/'docs/PLAN.md';p.write_text(update_plan(p.read_text()))

def commit(message):
 def git(*args,env=None,input=None,check=True):return subprocess.run(['git',*args],cwd=R,env=env,input=input,text=True,capture_output=True,check=check)
 def blob(text):return git('hash-object','-w','--stdin',input=text).stdout.strip()
 m,q,state,r,v=facts();history=state.get('history',[])+[x for x in (state.get('scored'),state.get('frozen')) if x]
 receipts=[x[k] for x in history for k in ('receipt','scored_receipt') if x.get(k)];jobs={x[k] for x in history for k in ('job','scored_job') if x.get(k)}
 shipped={m['shipped_source']}|{json.loads(p.read_text())['shipped_source'] for p in (H/'stages').glob('p*/manifest.json')}
 paths=['problems/muse_decode_layer/dev/s22',*sorted(shipped)]+sum(([p,str(Path(p).with_suffix('.py'))] for p in receipts),[])
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
