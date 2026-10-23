"""Official local tests and nearby fresh graphs with both FP8 paths forced on every test."""
import sys,gc,json
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
weights=ref.generate_weights(ref.WEIGHTS_SEED)
problem=le.Problem(ref,weights)
tests=le.tests_of(Path('problems/muse_prefill_layer/TASK.md').read_text())
all_results={}
for filename in sys.argv[1:]:
  m=le.load_module(Path(filename).stem,Path(filename))
  side=le.Side('submission',m,weights)
  results=le.run_tests(problem,side,tests,ref.WEIGHTS_SEED+123)
  all_results[filename]=results
  assert all(r['ok'] for r in results),results
  for ci,case in enumerate([dict(batch=1,ctx=1025,layer='full',q_len=257,page=64,kv_scale=1/64,spread='agentic'),
                           dict(batch=2,ctx=2053,layer='sliding',q_len=139,page=1,spread='agentic')]):
    data=problem.make_input(case,61370+ci)
    graph,out=le.capture(side.call,data)
    for j in range(2):
      fresh=problem.make_input(case,71370+ci*10+j);le.copy_into(data,fresh)
      graph.replay();torch.cuda.synchronize()
      ok,msg=problem.check(data,out)
      print(filename,'fresh graph',case,j,ok,msg,flush=True)
      assert ok,msg
    del graph,out,data,fresh
  del side,m
  gc.collect();torch.cuda.empty_cache()
suffix='256' if len(sys.argv)==2 and '256' in sys.argv[1] else '128'
Path(f'problems/muse_prefill_layer/dev/attention_fp8_validate_forced{suffix}.json').write_text(json.dumps(all_results,indent=2))
