"""Send the screened immutable oracle batch through h200_lab at the actual quota reset."""
import hashlib,json,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import h200_lab
p=R/'prof/lab/moe_p3a_operand.py';s=p.read_text();digest=hashlib.sha256(s.encode()).hexdigest()
for name in ('local_validation.json','leaf_validation.json'):
 v=json.loads((D/name).read_text());assert v['payload_sha256']==digest
assert not (D/'queue.json').exists(),'Do not duplicate a queued run'
host='muse_moe_w4a16';budget=700.;reset='2026-10-27T09:00:00+09:00'
entry=h200_lab.embed(p);assert entry==(p.stem,digest[:16],s)
wrapper,_=h200_lab.build([entry],host,phase='import',import_budget=budget,lab_only=True)
sha=hashlib.sha256(wrapper.encode()).hexdigest();out=R/'runs/screen/p3a_operand/wrapper.py';out.write_text(wrapper)
subprocess.run([sys.executable,str(R/'tools/local_eval.py'),str(out),'--tier','wrapper','--transport','lab-only','--json',str(out.with_name('transport.json'))],cwd=R,check=True,stdout=subprocess.DEVNULL)
wire=json.loads((D/'local_validation.json').read_text())['wire_bytes']
note=f'ExactS19 matched8seeds4symmetric all-round oracle:prelaid two-plane shared BF16 fragments,pack-free register floor,shorter Bf lifetime,32-HMMA accumulator recurrence with matched live sinks,useful-N HMMA instruction budget with matched decode sinks,L1/L2/L3 epilogue pressure,BASE/AA/shared-reservation controls.15arms1920times+allNVML/CTAwork;2CTA/zero spills all56staticrows.448nativecontrolTASK+162nearby/local-start/changed-capture LAZY checks pass;exact{wire}B local packet fits11700B. All oracle outputs ignored,no leaf-only P4 nomination. Largest buildable term chooses correct S21 production-graph build;whole six-load R decides,not per-call transfer.'
decision='split small-M MMA/activation residual into fragment loads,register lifetime,accumulator recurrence and N-padding'
log=R/'runs/detached'/f'{sha[:16]}.log';log.parent.mkdir(exist_ok=True)
cmd=[sys.executable,'tools/h200_lab.py','send',str(p.relative_to(R)),'--host',host,'--lab-only','--import-budget',str(budget),'--phase','import','--check','--not-before',reset,'--lane','P3a','--decision',decision,'--note',note]
with log.open('w') as f:proc=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(state='queued for quota reset;no H200 receipt yet',sender_pid=proc.pid,host=host,mode='check',not_before=reset,job=None,payload=str(p.relative_to(R)),payload_sha256=digest,wrapper_sha256=sha,expected_source=f'problems/{host}/lab/lab_{p.stem}.py',log=str(log.relative_to(R)),lane='P3a',decision=decision,command=cmd)
(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n');print('QUEUED',proc.pid,sha,digest)
