"""Reconcile profiler residence with stage completion increments; no new H200 claim."""
import json,re
from pathlib import Path
R=Path(__file__).resolve().parents[4]
x=json.loads((R/'problems/muse_decode_layer/dev/levers/current.json').read_text())
out={}
for load in ('decode 16','decode 32'):
 rows=[r for r in x['rows'] if r['case']['load']==load]
 weighted=sum(r['case']['weight']*r['stages_us']['boundaries'] for r in rows)
 excess=next(r for r in x['excess_1_3_by_stage'] if r['stage']=='boundaries')['per_load_us'][load]
 out[load]=dict(profile_boundary_residence_weighted_us=weighted,profile_excess_weighted_us=excess,
  profile_floor_per_layer_us=rows[0]['floors_us']['boundaries'],per_layer_residence_us=weighted/62,
  warning='node residence includes PDL dependency waits; not additive critical-path work')
timeline=(R/'problems/muse_decode_layer/dev/timeline/README.md').read_text()
for load,cell in out.items():
 b=load.split()[-1];rows={}
 for case,stage,span,inc,read in re.findall(r'\| (d'+b+r'[SF]) \| (b[12]) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)',timeline):
  rows.setdefault(case,[]).append(dict(stage=stage,span_us=float(span),increment_us=float(inc),read_MB=float(read)))
 assert len(rows)==2 and all(len(v)==2 for v in rows.values())
 cell['timeline']=rows
 cell['timeline_weighted_span_us']=sum((46 if k.endswith('S') else 16)*sum(r['span_us'] for r in v) for k,v in rows.items())
 cell['timeline_weighted_increment_us']=sum((46 if k.endswith('S') else 16)*sum(r['increment_us'] for r in v) for k,v in rows.items())
 cell['timeline_increment_per_layer_us']=cell['timeline_weighted_increment_us']/62
 cell['models_use_different_fixtures']=True
print(json.dumps(out,indent=2))
(Path(__file__).with_name('s11_accounting.json')).write_text(json.dumps(out,indent=2)+'\n')
