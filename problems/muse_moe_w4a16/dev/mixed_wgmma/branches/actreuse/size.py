"""Logical global activation requests and exact native resource admission."""
import json,math,re,runpy
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;n=runpy.run_path(str(H/'act_payload.py'));rows=[]
for seed in range(16):
 w=n['dc_weights'](29173+104729*seed,'cuda')
 for m in (128,256):
  for d in n['dc_fixtures'](w,m):
   c=n['act_counts'](d);cc=c['counts'];per=[]
   for e,tokens in enumerate(cc):
    low=0
    if 0<tokens<=8:
     for nt in (512,256):
      U=sum(0<v<=8 for v in cc)*nt;begin=sum(0<v<=8 for v in cc[:e])*nt
      low+=tokens*8192*sum(U*j//264<U*(j+1)//264 and U*(j+1)//264>begin and U*j//264<begin+nt for j in range(264))
    base=tokens*96*8192 if tokens>16 else tokens*192*8192 if tokens>8 else low
    a={k:base for k in n['ACT_NAMES']}
    if tokens>16:
     pad=(tokens//64*64+(32 if 0<tokens%64<=32 else 64 if tokens%64 else 0))*96*8192;g=tokens*8192*2;a['PACK']+=g
     for k,cs in [('TMA1',1),('MC2',2),('MC4',4)]:a[k]=pad/cs+g
     for k in ('W4N32','W4N64'):a[k]=base/2
    per.append(dict(expert=e,tokens=tokens,activation_bytes=a))
   assert all(abs(sum(x['activation_bytes'][k] for x in per)/1e6-c['activation_MB'][k])<1e-8 for k in n['ACT_NAMES'])
   c['per_expert']=per;c.update(M=m,seed=seed,fused8_CTA_count=sum(math.ceil(v/8) for v in cc),fused8_weight_MB=sum(math.ceil(v/8) for v in cc)*28.311552,fused32_ACT_KiB=256,fused8_ACT_KiB=64);rows.append(c)
 del w;torch.cuda.empty_cache()
(H/'counts.json').write_text(json.dumps(rows,indent=2)+'\n')
for m in (128,256):
 rr=[r for r in rows if r['M']==m]
 print(m,{a:sum(r['activation_MB'][a] for r in rr)/len(rr) for a in n['ACT_NAMES']},'fuse8 CTA/weightMB',sum(r['fused8_CTA_count'] for r in rr)/len(rr),sum(r['fused8_weight_MB'] for r in rr)/len(rr))
