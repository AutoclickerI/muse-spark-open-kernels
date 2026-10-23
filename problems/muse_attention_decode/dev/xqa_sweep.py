import sys,json,statistics
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
import local_eval as E
import torch
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
ref=E.load_module('xqa_ref',root/'reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None
mod=E.load_module('xqa_candidate',root/'cand_xqa.py')
scalar=E.load_module('xqa_scalar',root/'cand_xqa_p8.py')
timer=E.Timer()
rows=[]
for batch in (1,16,32):
 for layer in ('full','sliding'):
  data=ref.generate_input(batch=batch,ctx=80000,heads='tp8',layer=layer,q_len=8,page=1,spread='agentic' if batch>1 else 'equal',dist='coherent' if layer=='full' else 'soft',seed=43)
  results=[]
  for family,m in [('matrix',mod),('scalar',scalar)]:
   for p8 in (False,True):
    for stages in (3,5):
     for waves in (1,2,4):
      m.X_P8=p8;m.X_STAGES=stages;m.X_WAVES=waves
      graph,out=E.capture(m._xqa_verify,data)
      us=statistics.median(timer.replay_us(graph) for _ in range(5))-timer.fixed_us
      results.append((round(us,2),family,p8,stages,waves))
  results.sort()
  row={'batch':batch,'layer':layer,'variants':results};rows.append(row)
  print(json.dumps(row),flush=True)
  (root/'dev/xqa_sweep.json').write_text(json.dumps(rows,indent=2))
