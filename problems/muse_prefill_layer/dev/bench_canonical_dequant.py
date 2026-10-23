"""Local-only cold-L2 raw vs canonical dequant, with sequential scratch aliasing."""
from pathlib import Path
import sys,json
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from nvfp4 import dequantize
P=Path('problems/muse_prefill_layer')
m=le.load_module('dequant_bench',P/'cand_expert_portfolio.py')
w=ref.generate_weights(ref.WEIGHTS_SEED);st=m.prepare(w)
counts=torch.ones(32,device='cuda',dtype=torch.int32)
work=torch.empty((32,8192,4096),device='cuda',dtype=torch.bfloat16)
w2=work.view(-1)[:32*4096*4096].view(32,4096,4096)
flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
def packed():
    m._canonical_expand[(32,512)](st['w13p'],st['w13p'].view(torch.int32),work.view(torch.int32),counts,ROWS=8192,GATED=True,BLOCK=4096,num_warps=8)
    m._canonical_expand[(32,256)](st['w2p'],st['w2p'].view(torch.int32),w2.view(torch.int32),counts,ROWS=4096,GATED=False,BLOCK=4096,num_warps=8)
def raw():
    m._dequant_kernel[(32,1024)](w['w13'],w['w13_sf'].view(torch.uint8),work,counts,ROWS=8192,K=4096,BR=8,BKB=512,num_warps=4)
    m._dequant_kernel[(32,512)](w['w2'],w['w2_sf'].view(torch.uint8),w2,counts,ROWS=4096,K=4096,BR=8,BKB=512,num_warps=4)
records=[]
for name,fn in (('canonical',packed),('raw_v11',raw)):
    for _ in range(2):fn()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):fn()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    samples=[]
    for _ in range(5):
        flush.fill_(1);start.record();graph.replay();end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000)
    row=dict(name=name,local_sm120_us=sorted(samples)[2],samples_us=samples)
    print(row,flush=True);records.append(row)
    del graph
# Check complete expert tensors before norm/GS; both stages must be exact.
for gated,rows,bank,key in ((True,8192,st['w13p'],'w13'),(False,4096,st['w2p'],'w2')):
    out=work if gated else w2
    m._canonical_expand[(32,rows//16)](bank,bank.view(torch.int32),out.view(torch.int32),counts,ROWS=rows,GATED=gated,BLOCK=4096,num_warps=8)
    expected=dequantize(w[key][0],w[key+'_sf'][0],1.).to(torch.bfloat16)
    assert torch.equal(out[0],expected)
    print(key,'full expert BF16 bit-exact PASS',flush=True)
(P/'dev/dequant_benchmark_local.json').write_text(json.dumps(dict(hardware=torch.cuda.get_device_name(),cc='sm120',timings=records,full_expert_exact=True),indent=2))
