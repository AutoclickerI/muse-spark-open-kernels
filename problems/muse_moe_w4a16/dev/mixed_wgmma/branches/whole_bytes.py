"""Account for the changed activation traffic in the shared WGMMA recipe.

Same fixture seeds as the H200 diagnostic. Local router ties may differ;
these are address-count estimates, never hardware DRAM transactions.
"""
import json
from pathlib import Path
import runpy
import statistics
import torch
H=Path(__file__).resolve().parent
ns=runpy.run_path(str(H/'shared_payload.py'))
rows=[]
for seed in range(12):
 w=ns['dc_weights'](5353+7919*seed,'cuda')
 for m in (128,256):
  for fi,d in enumerate(ns['dc_fixtures'](w,m)):
   ids=d['topk_ids'].reshape(-1);n=torch.bincount(ids[(ids>=0)&(ids<32)].long(),minlength=32).cpu().tolist()
   light=sum(v for v in n if v<=16);heavy=sum(v for v in n if v>16)
   rows.append(dict(m=m,seed=seed,fixture=fi,light_tokens=light,heavy_tokens=heavy,
                    shared_activation_MB=(light+heavy)*8192*192/1e6,
                    added_light_activation_MB=light*8192*192/1e6,
                    shared_weight_MB=sum((v+31)//32 for v in n)*4096*4096*3*.5625/1e6,
                    unique_weight_MB=sum(v>0 for v in n)*4096*4096*3*.5625/1e6))
 del w;torch.cuda.empty_cache()
summary={m:{k:statistics.mean(r[k] for r in rows if r['m']==m) for k in rows[0] if k not in ('seed','fixture','m')} for m in (128,256)}
(H/'whole_bytes.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');print(summary)
