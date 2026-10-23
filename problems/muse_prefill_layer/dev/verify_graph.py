import sys, gc
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
weights=ref.generate_weights(ref.WEIGHTS_SEED)
problem=le.Problem(ref,weights)
cases=[dict(batch=1,ctx=1025,layer='full',q_len=257,page=64,kv_scale=1/64,spread='agentic'),
       dict(batch=2,ctx=2053,layer='sliding',q_len=139,page=1,spread='agentic')]
for filename in sys.argv[1:]:
 mod=le.load_module(Path(filename).stem,Path(filename))
 side=le.Side('submission',mod,weights)
 for ci,case in enumerate(cases):
  data=problem.make_input(case,41902+ci)
  graph,out=le.capture(side.call,data)
  for i in range(2):
   fresh=problem.make_input(case,51902+ci*10+i)
   le.copy_into(data,fresh)
   graph.replay(); torch.cuda.synchronize()
   ok,msg=problem.check(data,out)
   print(filename,case,'fresh',i,ok,msg,flush=True)
   assert ok,msg
  del graph,out,data,fresh
 del side,mod
 gc.collect(); torch.cuda.empty_cache()
