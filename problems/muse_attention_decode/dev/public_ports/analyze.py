"""No filtering: symmetric cold rounds -> paired seed estimates and t7 intervals."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import paired_evidence
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();r=json.loads(a.receipt.read_text());d=paired_evidence(r['result'].get('log_tail',''),'RESULT')
assert d and d['complete'] and not d['failures'],d
assert d['checks']==dict(official=128,ABI=192,scored=384),d['checks']
manifest=json.loads((H/'out/lab_manifest.json').read_text());assert d['seeds']==manifest['seeds']
assert d['identities']==dict(source=manifest['base_sha256'],**manifest['identities'])
assert len(d['round_ns'])==12 and all(len(samples)==8 for samples in d['round_ns'].values())
fixed=d['fixed_ns'];points=[];controls={};times={};selection={};all_pieces={}
def ci(v):
 mean=statistics.mean(v);se=statistics.stdev(v)/len(v)**.5
 return dict(r=mean,se=se,lo=mean-2.364624251*se,hi=mean+2.364624251*se,seed_ratios=v)
for cell,samples in d['round_ns'].items():
 assert all(set(s)=={'A','A2','O','L'} and all(len(v)==8 and min(v)>fixed for v in s.values()) for s in samples)
 means=[{k:statistics.mean(v)-fixed for k,v in s.items()} for s in samples];times[cell]=means
 controls[cell]=ci([m['A2']/m['A'] for m in means]);best=None
 for arm in ('O','L'):
  p=dict(cell=cell,arm=arm,**ci([m[arm]/m['A'] for m in means]));p['nominated']=p['r']<1;points.append(p)
  if p['nominated']:
   all_pieces.setdefault(arm,{})[cell]=arm
   if best is None or p['r']<best[0]:best=(p['r'],arm)
 if best:selection[cell]=best[1]
 print(cell,*(arm+':'+f'{q["r"]:.6f}[{q["lo"]:.6f},{q["hi"]:.6f}]' for arm in ('O','L','AA') for q in ([next(p for p in points if p['cell']==cell and p['arm']==arm)] if arm!='AA' else [controls[cell]])))
result=dict(measured=True,job=r.get('id',r.get('job',r.get('job_id'))),receipt=str(a.receipt.resolve().relative_to(R)),payload_sha256=manifest['payload_sha256'],seeds=d['seeds'],checks=d['checks'],controls=controls,points=points,selection=selection,seed_ns=times,raw_rounds_sha256=hashlib.sha256(json.dumps(d['round_ns'],sort_keys=True).encode()).hexdigest(),elapsed=d['elapsed'],graph_nodes=d['graph_nodes'],gate='Every correct point<1 nominated per cell; P4 composes and confirms held-out whole-layer G0. A:A drift disclosed, no per-load veto.')
(H/'result.json').write_text(json.dumps(result,indent=2)+'\n');(H/'receipt_data.json').write_text(json.dumps(d,indent=2)+'\n');(H/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
for arm,sel in all_pieces.items():(H/(arm+'_selection.json')).write_text(json.dumps(sel,indent=2)+'\n')
