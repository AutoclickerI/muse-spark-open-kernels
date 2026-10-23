from pathlib import Path
import sys,json,statistics
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root.parents[1]/'tools')]
import local_eval as E
ref=E.load_module('warp_bench_ref',root/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**k:None
mods={name:E.load_module('warp_bench_'+name,root/f) for name,f in [('pipeline','cand_xqa_pipeline.py'),('warp','dev/xqa_warp_candidate.py')]}
timer=E.Timer();results=[]
for batch,layer in [(16,'full'),(32,'full'),(1,'sliding'),(16,'sliding'),(32,'sliding')]:
 data=ref.generate_input(batch=batch,ctx=80000,heads='tp8',q_len=8,layer=layer,page=1,spread='agentic',dist='coherent' if layer=='full' else 'soft',seed=923)
 rows=[]
 for name,m in mods.items():
  for p8,st in [(False,3),(False,4),(True,3),(True,4)]:
   if name=='pipeline' and (p8 or st!=3):continue
   for waves in (1,2,4,8):
    m.X_P8,m.X_STAGES,m.X_WAVES=p8,st,waves
    g,out=E.capture(m._xqa_verify,data)
    for _ in range(3):g.replay()
    torch.cuda.synchronize()
    ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
    us=statistics.median(timer.replay_us(g)-timer.fixed_us for _ in range(5))
    rows.append([round(us,2),name,p8,st,waves]);del g,out
  print('WARP_BENCH_PART',batch,layer,name,sorted(rows),flush=True)
 results.append(dict(batch=batch,layer=layer,variants=sorted(rows)))
 (root/'dev/xqa_warp_bench.json').write_text(json.dumps(results,indent=2))
 del data;torch.cuda.empty_cache()
