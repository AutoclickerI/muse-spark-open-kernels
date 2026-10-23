"""Exercise the actual combined V6+FP8 production branch in a fresh full-layer graph."""
import sys
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
filenames=[Path(x) for x in sys.argv[1:]] or [Path('problems/muse_prefill_layer/cand_canonical_attention.py')]
weights=ref.generate_weights(ref.WEIGHTS_SEED)
problem=le.Problem(ref,weights)
for filename in filenames:
  m=le.load_module(filename.stem,filename)
  side=le.Side('submission',m,weights)
  case=dict(batch=1,ctx=4096,layer='full',q_len=2048,page=64,kv_scale=1/64,spread='agentic')
  data=problem.make_input(case,89123)
  graph,out=le.capture(side.call,data)
  for seed in (99123,99124):
    fresh=problem.make_input(case,seed);le.copy_into(data,fresh)
    graph.replay();torch.cuda.synchronize()
    ok,msg=problem.check(data,out)
    print(filename,'actual V6+FP8 production branch',case,'freshseed',seed,ok,msg,flush=True)
    assert ok,msg
