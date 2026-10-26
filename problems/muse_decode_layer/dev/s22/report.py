"""Native S22 G0 and in-graph per-cell evidence; selection always needs new seeds."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import compose
H=compose.H;sys.path.insert(0,str(compose.R/'tools'))
from promotion import evaluate,interval
from lab_packet import paired_evidence
codec=compose.load(H.parent/'s21/evidence.py','s22_evidence')
def decode(raw):return codec.s21_decode_samples(codec.s21_unpack_evidence(raw))

def analyze(receipt_path,manifest_path=None,official=None):
 manifest_path=Path(manifest_path or H/'manifest.json');m=json.loads(manifest_path.read_text());rp=Path(receipt_path);receipt=json.loads(rp.read_text())
 assert hashlib.sha256(rp.with_suffix('.py').read_bytes()).hexdigest()==m['wrapper_sha256']
 evidence=paired_evidence(receipt['result']['log_tail'],'S22');stage=H/'stages'/f"p{m['phase']}";stage.mkdir(exist_ok=True,parents=True)
 if evidence is None:
  result=dict(job=receipt['id'],measured=False,lab_gate=False,reason='No complete paired archive',drop_cells=[],disable_hybrids=[],phase=m['phase'])
 else:
  raw=decode(evidence);official_pass=False
  if official:
   op=Path(official);assert hashlib.sha256(op.with_suffix('.py').read_bytes()).hexdigest()==m['shipped_sha256'];official_pass=json.loads(op.read_text())['result'].get('passed') is True
  full=evaluate(dict(id=receipt['id'],result=dict(passed=official_pass,gpu=receipt['result'].get('gpu'))),raw,m)
  prim=[full['arm_status'][n] for n in ('A','A2','INC')];prepare_gate=0<raw.get('host_prepare_s',float('inf'))<=300
  gate=bool(full['score']['measured'] and full['score']['CI95'][0]>1 and prepare_gate and all(not s['failures'] and s['nearby_task_checks']==s['nearby_expected_checks'] for s in prim))
  cells={};hybrids={};drops=[];disabled=[]
  for name,spec in m['arms'].items():
   if spec['kind'] not in ('S19_cell','prior_port'):continue
   label=spec['cell'];vectors=raw.get('cell_ratios',{}).get(label,{});scale=raw.get('cell_ratio_scale',1000000)
   measured=bool(full['pieces'][name]['measured'] and len(vectors.get(name,[]))==len(raw['seed_ids']) and len(vectors.get('A2',[]))==len(raw['seed_ids']))
   out=dict(measured=measured,arm=name,comparison='time(leave-out)/time(S22); >1 favors keeping',cell=label)
   if measured:
    ratios=[1+x/scale for x in vectors[name]];aa=[1+x/scale for x in vectors['A2']]
    out.update(interval(ratios,aa));out['AA']=interval(aa);out['loses']=out['CI95'][1]<1
    if out['loses']:(drops if spec['kind']=='S19_cell' else disabled).append(label)
   (cells if spec['kind']=='S19_cell' else hybrids)[label]=out
  batch=H/'batches'/m['wrapper_sha256'][:16];batch.mkdir(exist_ok=True,parents=True);(batch/'full_report.json').write_text(json.dumps(full,indent=2)+'\n')
  slim=lambda v:{k:x for k,x in v.items() if k not in ('seed_ids','selection_seeds','per_seed_R')}
  aa=[];alternate=[]
  if full['score']['measured']:
   for i in range(len(raw['seed_ids'])):
    control=math.exp(sum(math.log(c['samples'][i]['A2']) for c in raw['loads'].values())/6);primary=math.exp(sum(math.log(c['samples'][i]['INC']) for c in raw['loads'].values())/6);aa.append(control);alternate.append(primary/control)
  result=dict(job=receipt['id'],phase=m['phase'],source_sha256=m['sha256'],shipped_sha256=m['shipped_sha256'],measured=full['measured'],lab_gate=gate,freeze_gate=full['freeze_gate'],prepare_gate=prepare_gate,
   score=slim(full['score']),pieces={n:slim(v) for n,v in full['pieces'].items()},cells=cells,hybrids=hybrids,drop_cells=sorted(drops),disable_hybrids=sorted(set(disabled)-set(drops)),
   checks={k:raw[k] for k in ('complete','failures','task_checks','expected_task_checks','nearby_task_checks','nearby_expected_checks')},prepare={k:raw.get(k) for k in ('host_prepare_s','build_s','delta_s','phase_s')},
   measurement_protocol=raw['measurement_protocol'],control_diagnostics=dict(AA=interval(aa),INC_over_A2=interval(alternate)) if aa else None,
   full_archive=str((batch/'full_report.json').relative_to(compose.R)),selected_build_needs_new_confirmation=bool(drops or disabled))
 (stage/'results.json').write_text(json.dumps(result,indent=2)+'\n');(H/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 phases={p.parent.name:{k:v for k,v in json.loads(p.read_text()).items() if k in ('job','phase','source_sha256','cells','hybrids')} for p in sorted((H/'stages').glob('p*/results.json'))}
 (H/'P5B_FEEDBACK.json').write_text(json.dumps(dict(latest_phase=result['phase'],phases=phases),indent=2)+'\n')
 return result
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--manifest',type=Path);ap.add_argument('--official-task',type=Path);a=ap.parse_args();r=analyze(a.receipt,a.manifest,a.official_task);print(json.dumps(r,indent=2))
