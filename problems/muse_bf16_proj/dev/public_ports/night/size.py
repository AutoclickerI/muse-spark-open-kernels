"""Assign the existing small-M scenario to actual native S19 sites, not residence."""
import json,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'tools'))
from promotion import T95
src=json.loads((H.parents[1]/'native_boundary/sizing_results.json').read_text())
def ci(x):
 m=statistics.mean(x);d=T95[len(x)-1]*statistics.stdev(x)/math.sqrt(len(x));return [m,m-d,m+d]
rows=[]
for b in (16,32):
 for group,roles in [('QKV',('qkv3','qkv_gate')),('O',('o',)),('pool_router',('pool_router',)),('unpooler',('unpooler',)),('gate',('gate',))]:
  values=[0.]*8;shapes={}
  for window,count in ((2048,46),(0,16)):
   parts=[]
   for role in roles:
    rr=next((r for r in src['rows'] if r['case']==[b,1,window] and r['role']==role),None)
    if rr:
     v=[max(x-rr['roof_us'],0) for x in rr['critical_us']];values=[a+x*count/62 for a,x in zip(values,v)];parts.append(dict(role=role,encoding=rr['encoding'],bytes=rr['weight_bytes'],splits=rr['splits'],exposed_us=rr['critical_us_CI'],stream_us=rr['roof_us'],surplus_us=ci(v)))
   shapes['S' if window else 'F']=parts
  rows.append(dict(T=b,site=group,surplus_us=ci(values),samples=values,shapes=shapes))
out=dict(receipts=dict(native=src['job'],timeline='9c16c850f7b941a5'),R=src['terms']['small_M']['R'],rows=rows,limits='Existing completion-path scenario max(exposed-actual-bank fitted stream,0), not exclusive physical cycles or guaranteed gain. Fit4.61419582TB/s+2.876us. Timeline has a different max(bytes/FLOPs) roof and1.3 allowance; do not subtract or add those models.')
(H/'sizing.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# T16/T32 carriers of the small-M stream scenario','','Native e70024b1 and independent S19 timeline9c16c850. Values are weighted us/layer,46S+16F; not physical decomposition or promotion.','','| T | Site | Surplus us [sampling95%CI] |','|---:|---|---|']
for r in rows:
 v,l,h=r['surplus_us'];lines.append(f"| {r['T']} | {r['site']} | {v:.6f} [{l:.6f},{h:.6f}] |")
lines+=['','T32 pool/router is **dense BF16, six splits**,75,497,472bytes; T16 pool/router is NF12,S1,56,936,448bytes. QKV16 is raw BF16; QKV32 and both unpoolers are NF12. Gate has no positive exposed surplus in this model. O32 is only.008758us and remains the parent port point rather than a new build.','', 'Timeline9c16c850 weighted critical/roof: d16 pool27.06/15.84us, QKV9.85/8.15, O6.14/2.80, unpool14.89/10.68; d32 pool21.50/16.58,QKV11.99/6.38,O6.85/3.62,unpool15.75/10.87. Those byte/tensor roofs exclude the fitted intercept, so their excess differs.']
(H/'SIZING.md').write_text('\n'.join(lines)+'\n')
print([(r['T'],r['site'],r['surplus_us'][0]) for r in rows])
