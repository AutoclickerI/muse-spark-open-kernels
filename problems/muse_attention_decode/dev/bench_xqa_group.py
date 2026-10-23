from pathlib import Path
import sys,json,statistics
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root.parents[1]/'tools')]
import local_eval as E
m=E.load_module('next_bench',root/'dev/xqa_next_candidate.py')
ref=E.load_module('next_bench_ref',root/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**k:None
timer=E.Timer();rows=[]
for batch,layer in [(16,'full'),(32,'full'),(1,'sliding'),(16,'sliding'),(32,'sliding')]:
 data=ref.generate_input(batch=batch,ctx=80000,heads='tp8',q_len=8,layer=layer,page=1,spread='agentic',dist='coherent' if layer=='full' else 'soft',seed=9939)
 values=[]
 for family,stages in [('group',(5,7)),('warp',(3,))]:
  for st in stages:
   for waves in (1,2,4,8):
    m._next_set((family,st,waves));g,out=E.capture(m._xqa_verify,data)
    for _ in range(3):g.replay()
    torch.cuda.synchronize();ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
    us=statistics.median(timer.replay_us(g)-timer.fixed_us for _ in range(5))
    values.append([round(us,2),family,st,waves]);del g,out
 rows.append(dict(batch=batch,layer=layer,variants=sorted(values)))
 print('NEXT_BENCH',batch,layer,sorted(values),flush=True)
 (root/'dev/xqa_next_bench.json').write_text(json.dumps(rows,indent=2));del data;torch.cuda.empty_cache()
