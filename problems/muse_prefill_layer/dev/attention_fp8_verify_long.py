"""Fresh replay of complete 16K attention blocks, sampled FP32 rows attend every key."""
import json
from pathlib import Path
import torch
import attention_fp8_probe as p

results=[]
# The probe module's main loop is protected; importing it only loads helpers.
for prefix in (32768,65536):
  for contig in (False,True):
    data,q=p.data_for(1,16384,prefix,seed=12345)
    f=lambda:p.m._attention_fp8(data,q,{},contig=contig)
    f(); torch.cuda.synchronize()
    stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph): output=f()
    torch.cuda.current_stream().wait_stream(stream)
    for seed in (23456,34567):
      fresh,fq=p.data_for(1,16384,prefix,seed=seed,soft=(seed==34567))
      q.copy_(fq)
      for key in ('k_cache','v_cache','page_table','seq_lens','k_scale','v_scale'):
        data[key].copy_(fresh[key])
      graph.replay(); torch.cuda.synchronize()
      at,refs=p.ref_rows(data,q)
      for (b,idx),ref in zip(at,refs):
        got=output.view(1,16384,16,64)[b,idx].float()
        rel=float((got-ref).norm()/ref.norm())
        assert rel<.04,rel
        assert torch.isfinite(output).all()
        row=dict(prefix=prefix,contiguous=contig,seed=seed,soft=(seed==34567),relative_l2=rel)
        print(row,flush=True);results.append(row)
      del fresh,fq,refs
    del data,q,output,graph
    torch.cuda.empty_cache()
Path('problems/muse_prefill_layer/dev/attention_fp8_verify_long.json').write_text(json.dumps(results,indent=2))
