import json
from pathlib import Path
import torch
import attention_fp8_probe as p
variants=[dict(bq=16,bn=128,group=8,warps=8,stages=2,contig=True),
          dict(bq=16,bn=128,group=8,warps=16,stages=2,contig=True),
          dict(bq=8,bn=128,group=8,warps=8,stages=2,contig=True),
          dict(bq=8,bn=256,group=8,warps=8,stages=2,contig=True),
          dict(bq=16,bn=256,group=8,warps=8,stages=1,contig=True)]
flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
rows=[]
for prefix in (32768,65536):
  data,q=p.data_for(1,16384,prefix,seed=23457,soft=True)
  at,refs=p.ref_rows(data,q)
  for kw in variants:
    try:
      f=lambda:p.m._attention_fp8(data,q,{},**kw)
      out=f();torch.cuda.synchronize()
      err=max(float((out.view(1,16384,16,64)[b,idx].float()-ref).norm()/ref.norm()) for (b,idx),ref in zip(at,refs));assert err<.04
      stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
      with torch.cuda.stream(stream):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=f()
      torch.cuda.current_stream().wait_stream(stream)
      times=[]
      for _ in range(3):
        flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();z.record();z.synchronize();times.append(a.elapsed_time(z)*1000)
      row=dict(prefix=prefix,variant=kw,error=err,local_sm120_us=sorted(times)[1]);print(row,flush=True);rows.append(row)
      del graph,out
    except Exception as e: print('FAILED',kw,type(e).__name__,str(e),flush=True)
  del data,q,refs
  torch.cuda.empty_cache()
Path('problems/muse_prefill_layer/dev/attention_fp8_bench_tiles.json').write_text(json.dumps(rows,indent=2))
