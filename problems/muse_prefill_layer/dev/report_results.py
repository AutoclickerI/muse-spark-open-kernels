import json, re
from pathlib import Path
from collections import defaultdict
root=Path(__file__).resolve().parents[3]
rows=[]
for f in sorted((root/'runs/muse_prefill_layer').glob('*.json')):
 d=json.loads(f.read_text()); r=d.get('result',{}); loads=defaultdict(lambda:[0.,0.])
 for rec in r.get('records',[]):
  if rec.get('event')=='benchmark' and rec.get('status')=='pass':
   c=rec['case']; w=c.get('weight',1)
   loads[c['load']][0]+=rec['baseline_ns']*w
   loads[c['load']][1]+=rec['median_ns']*w
 row=dict(file=str(f.relative_to(root)),job=d['id'],mode=d['mode'],passed=r.get('passed'),speedup=r.get('speedup'),
          loads={k:round(a/b,6) for k,(a,b) in loads.items()},error=r.get('error'))
 rows.append(row)
 print(json.dumps(row))
 for rec in r.get('records',[]):
  if rec.get('event')!='profile':continue
  c=rec['case']; st=defaultdict(float)
  for line in rec['report'].splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs',line)
   if not m:continue
   name,us=m.group(1),float(m.group(2))
   if name=='total':st['total']=us;continue
   if any(s in name for s in ['Marlin','moe_gemm','_dequant','deep_gemm','_swiglu','_combine','_gather_norm','_pre_expert_norm','_post_expert','act_and_mul','combine_tiled','gather_tiled','humming']):kind='experts'
   elif 'flash::' in name or '_gather_cache' in name:kind='attention'
   elif name.startswith('nvjet') or name.startswith('ampere') or 'gemm' in name:kind='dense'
   else:kind='glue'
   st[kind]+=us
  print(' profile',c['load'],c['layer'],c['ctx'],dict(st))
Path(__file__).with_name('results_summary.json').write_text(json.dumps(rows,indent=2))
