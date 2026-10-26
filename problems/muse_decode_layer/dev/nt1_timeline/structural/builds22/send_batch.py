"""One quota-reset paired run, after the older same-rank P3a diagnostic."""
import hashlib,json,subprocess,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import h200_lab
def send(replace=False):
 m=json.loads((D/'lab_manifest.json').read_text());c=json.loads((D/'manifest.json').read_text());leaf=json.loads((D/'leaf_validation.json').read_text());screens=json.loads((D/'screen_validation.json').read_text())
 if not replace:assert not (D/'queue.json').exists(),'Do not duplicate the paired send'
 for name in m['active']:
  assert leaf[name]['all_pass'] and leaf[name]['source_sha256']==c[name]['candidate_sha256']
  assert screens[name]['shipped_sha256']==c[name]['shipped_sha256'] and all(screens[name][k]['passed'] for k in ('normal','forced'))
 if not replace:
  v=json.loads((D/'rehearsal_validation.json').read_text());assert v['payload_sha256']==m['payload_sha256'] and v['complete'] and v['local_cannot_promote']
 p=R/m['payload'];entry=h200_lab.embed(p);assert entry[1]==m['payload_sha256'][:16]
 source,_=h200_lab.build([entry],m['host'],phase='import',import_budget=850.,lab_only=True);assert hashlib.sha256(source.encode()).hexdigest()==m['wrapper_sha256']
 note=f"P3a conditional builds on frozen S22 4b8c803d: {','.join(m['active'])};raw/shared fragment lifetime,independent accumulator chains,singleton SIMT without padded MMA,four-warp epilogue. All changes correct/native or explicit CACHE1 local bank stand-in;1350leaf gates,all normal+forced13TASK/12graphs,sm90a2CTA/zero spills. One64freshseed x16symmetric all-round whole-production-graph cohort,all12shapes allarms vs exact S22+A:A;5376TASK+780nearby before pruning. Six-load T46/16 throughpromotion.py;no per-call gain transfer. Older P3a diagnostic e1710fb3 precedes this same-rank send;finite consumer may prune only before this waiting send dispatches. No scored submission."
 cmd=[sys.executable,'tools/h200_lab.py','send',m['payload'],'--host',m['host'],'--lab-only','--phase','import','--import-budget','850','--check','--not-before','2026-10-27T09:00:00+09:00','--lane','P3a','--decision','S22 whole-graph test of every plausible P3a operand-diagnostic outcome','--note',note]
 log=R/'runs/detached'/(m['wrapper_sha256'][:16]+'.launch.log');log.parent.mkdir(exist_ok=True)
 with log.open('w') as out:subprocess.run(cmd,cwd=R,stdout=out,stderr=subprocess.STDOUT,check=True)
 waiting=[json.loads(p.read_text()) for p in (R/'runs/waiting').glob('*.json')];w=next((v for v in waiting if v['sha256']==m['wrapper_sha256']),None)
 if w is None:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:args=(p/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if len(args)>3 and args[1].endswith(b'tools/okc_run.py') and b'problems/muse_moe_w4a16/lab/lab_moe_p3a_s22.py' in args:w=dict(pid=int(p.name));break
 assert w is not None,'No actual paired sender found'
 q=dict(sender_pid=w['pid'],host=m['host'],mode='check',wrapper_sha256=m['wrapper_sha256'],payload_sha256=m['payload_sha256'],active=m['active'],expected_source='problems/muse_moe_w4a16/lab/lab_moe_p3a_s22.py',waiting=f"runs/waiting/{w['pid']}.json",not_before='2026-10-27T09:00:00+09:00',state='queued behind diagnostic;no judge job assigned',job=None,decision=cmd[cmd.index('--decision')+1])
 (D/'queue.json').write_text(json.dumps(q,indent=2)+'\n');print('PAIRED_QUEUED',q,flush=True);return q
if __name__=='__main__':send()
