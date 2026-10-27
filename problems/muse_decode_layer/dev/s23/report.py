"""Native S23 G0 over S22 phase 1 and per-group leave-out R (>1 favors keeping the group)."""
import hashlib,json,sys
from pathlib import Path
import compose
H=compose.H;R=compose.R;sys.path.insert(0,str(R/'tools'))
from promotion import evaluate
from lab_packet import paired_evidence
codec=compose.load(H.parent/'s21/evidence.py','s23_evidence')
def decode(raw):return codec.s21_decode_samples(codec.s21_unpack_evidence(raw))

def analyze(receipt_path):
 m=json.loads((H/'manifest.json').read_text());rp=Path(receipt_path);receipt=json.loads(rp.read_text())
 assert hashlib.sha256(rp.with_suffix('.py').read_bytes()).hexdigest()==m['wrapper_sha256']
 evidence=paired_evidence(receipt['result']['log_tail'],'S23');stage=H/'stages'/f"p{m['phase']}";stage.mkdir(exist_ok=True,parents=True)
 if evidence is None:result=dict(job=receipt['id'],phase=m['phase'],measured=False,lab_gate=False,reason='No complete paired archive')
 else:
  raw=decode(evidence)
  # Lab-only hosts fail setup after the import-time lab by design: TASK_pass stays false here and the
  # official TASK result comes from the scored run. lab_gate uses the in-lab TASK and nearby checks.
  full=evaluate(dict(id=receipt['id'],result=dict(passed=False,gpu=receipt['result'].get('gpu'))),raw,m)
  prim=[full['arm_status'][n] for n in ('A','A2','INC')];prepare_gate=0<raw.get('host_prepare_s',float('inf'))<=300
  gate=bool(full['score']['measured'] and full['score']['CI95'][0]>1 and prepare_gate and all(not s['failures'] and s['nearby_task_checks']==s['nearby_expected_checks'] for s in prim))
  slim=lambda v:{k:x for k,x in v.items() if k not in ('seed_ids','selection_seeds','per_seed_R')}
  (stage/'full_report.json').write_text(json.dumps(full,indent=2)+'\n')
  result=dict(job=receipt['id'],phase=m['phase'],source_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],measured=full['measured'],lab_gate=gate,prepare_gate=prepare_gate,
   score=slim(full['score']),groups={m['arms'][n]['group']:slim(v) for n,v in full['pieces'].items()},
   checks={k:raw[k] for k in ('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks')},prepare={k:raw.get(k) for k in ('host_prepare_s','build_s','delta_s','phase_s')},
   measurement_protocol=raw['measurement_protocol'],full_archive=str((stage/'full_report.json').relative_to(R)))
 (stage/'results.json').write_text(json.dumps(result,indent=2)+'\n');(H/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(analyze(sys.argv[1]),indent=2))
