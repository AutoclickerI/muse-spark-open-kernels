"""Reproduce dense-vs-glue accounting from immutable H200 receipts."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[4]
JOB='a30fb411b0c74ffb'
receipt=ROOT/'runs/muse_prefill_layer/20261023T224004Z_submit_a30fb411b0c74ffb.json'
raw=json.loads(receipt.read_text())
line=next(l for l in raw['result']['log_tail'].splitlines() if l.startswith('SCORE_COMPLETE_TABLE '))
table=json.loads(line.split(' ',1)[1])
roles={'qkv':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),
       'unpooler':(8192,4096),'router_cat':(512,8192)}
rows=[]
for role,(n,k) in roles.items():
    r=table['stages']['dense'][str((role,16384))]
    catalog=table['catalogs'][r['catalog']]
    us=r['us'][catalog.index(r['selected'])]
    flops=2*16384*n*k
    floor_us=flops/(989*1e6)
    rows.append(dict(role=role,choice=r['selected'],us=us,tensor_floor_us=floor_us,
                     tflops=flops/us/1e6,percent_peak=100*floor_us/us,
                     weighted_ms=us*.186,excess_ms=(us-floor_us)*.186))
profile=json.loads((ROOT/raw['profile']['json']).read_text())
actual=[]
for p in profile['profiles']:
    if p['case']['load']!='first turn':continue
    known={}
    for line in p['report'].splitlines():
        m=re.match(r'^(.*?)\s+([0-9.]+) µs\s+[0-9.]+%\s+([0-9]+)\s*$',line)
        if not m:continue
        name,us,count=m.group(1),float(m.group(2)),int(m.group(3))
        if name.startswith('nvjet_'):known['dense_gemm_us']=known.get('dense_gemm_us',0)+us
        if name in ('_boundary_kernel','_canonical_expand_coalesced','_routing_kernel'):
            known[name]=dict(total_us=us,calls=count)
    actual.append(dict(case=p['case'],known=known))
report=dict(job=JOB,classification='H200 receipts; selected tuner microbenchmarks vs separate profile calls',
    rows=rows,weighted_selected_dense_ms=sum(r['weighted_ms'] for r in rows),
    weighted_dense_tensor_floor_ms=sum(r['tensor_floor_us']*.186 for r in rows),
    weighted_profile_dense_ms=sum(x['known']['dense_gemm_us']*x['case']['weight']/1000 for x in actual),
    weighted_profile_boundary_ms=sum(x['known']['_boundary_kernel']['total_us']*x['case']['weight']/1000 for x in actual),
    weighted_profile_dequant_ms=sum(x['known']['_canonical_expand_coalesced']['total_us']*x['case']['weight']/1000 for x in actual),
    profiles=actual)
out=ROOT/'problems/muse_prefill_layer/dev/dense_a30/decomposition.json'
out.write_text(json.dumps(report,indent=2)+'\n')
print('| Role | selected us | tensor floor us | TFLOPS | peak % | first-turn excess ms |')
print('|---|---:|---:|---:|---:|---:|')
for r in rows:
    print(f"| {r['role']} | {r['us']:.2f} | {r['tensor_floor_us']:.2f} | {r['tflops']:.1f} | {r['percent_peak']:.1f} | {r['excess_ms']:.2f} |")
for k,v in report.items():
    if k.startswith('weighted_'):print(k,round(v,3))
