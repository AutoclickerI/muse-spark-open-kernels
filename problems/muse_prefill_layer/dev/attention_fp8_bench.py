"""Local sm120 cold-L2 graph comparison, never an H200 performance claim."""
import json
from pathlib import Path
import torch,triton
import attention_fp8_probe as p

def bf16(data,q):
  cfg=data['config'];B,L=cfg['batch'],cfg['q_len']
  out=torch.empty((B*L,1024),device=q.device,dtype=torch.bfloat16)
  p.m._attn_fwd_kernel[(triton.cdiv(L,16),B*2)](q,data['k_cache'],data['v_cache'],data['page_table'],
    data['seq_lens'],out,data['k_scale'],data['v_scale'],cfg['sm_scale']*cfg['qk_scale_factor'],
    L,data['page_table'].shape[1],PAGE=cfg['page_size'],WINDOW=cfg['window'],BQ=16,BN=64,num_warps=8,num_stages=2)
  return out
variants=[('bf16_triton',{}),('fp8_direct',dict()),('fp8_contig',dict(contig=True)),
          ('fp8_direct_bn256',dict(bn=256)),('fp8_contig_bn256',dict(contig=True,bn=256))]
flush=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8)
results=[]
for L,S in ((16384,32768),(16384,65536)):
  data,q=p.data_for(1,L,S,seed=8231)
  for name,kw in variants:
    f=(lambda:bf16(data,q)) if name=='bf16_triton' else (lambda:p.m._attention_fp8(data,q,{},**kw))
    f();torch.cuda.synchronize()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):output=f()
    torch.cuda.current_stream().wait_stream(stream)
    times=[]
    for _ in range(5):
      flush.zero_()
      a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
      a.record();graph.replay();z.record();z.synchronize()
      times.append(a.elapsed_time(z)*1000)
    row=dict(L=L,prefix=S,variant=name,local_sm120_us=sorted(times)[2],samples_us=times)
    print(row,flush=True);results.append(row)
    del graph,output
  del data,q
  torch.cuda.empty_cache()
Path('problems/muse_prefill_layer/dev/attention_fp8_bench.json').write_text(json.dumps(results,indent=2))
