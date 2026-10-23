import sys,json
from pathlib import Path
sys.path[:0]=['problems/muse_prefill_layer/dev','problems/muse_prefill_layer']
import torch
import canonical_dequant as d
from moe_canonical_host import repack_v6
from nvfp4 import dequantize
records=[]
def dense(bank,counts,rows,gated,warps,out=None):
    if out is None:out=torch.empty((bank.shape[0],rows,4096),device=bank.device,dtype=torch.bfloat16)
    d._canonical_expand_coalesced[(bank.shape[0],rows//16,2)](bank,bank.view(torch.int32),out.view(torch.int32),counts,ROWS=rows,GATED=gated,KB=2048,num_warps=warps)
    return out

for rows,gated in ((32,True),(16,False)):
    raw=torch.arange(rows*2048,device='cuda').remainder(256).to(torch.uint8).view(1,rows,2048)
    sb=torch.arange(rows*256,device='cuda').remainder(254)
    sb=torch.where(sb>=127,sb+1,sb).to(torch.uint8).view(1,rows,256)
    sf=sb.view(torch.float8_e4m3fn)
    bank=repack_v6(raw,sf,gated)
    counts=torch.ones(1,device='cuda',dtype=torch.int32)
    expected=dequantize(raw[0],sf[0],1.).to(torch.bfloat16)[None]
    for warps in (4,8):
        out=dense(bank,counts,rows,gated,warps)
        torch.cuda.synchronize()
        mismatch=int((out!=expected).sum())
        print(rows,gated,warps,'mismatches',mismatch,flush=True)
        assert mismatch==0
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):dense(bank,counts,rows,gated,warps,out=out)
        out.fill_(2);counts.zero_();graph.replay();torch.cuda.synchronize();assert (out==2).all()
        counts.fill_(1);graph.replay();torch.cuda.synchronize();assert torch.equal(out,expected)
        records.append(dict(rows=rows,gated=gated,warps=warps,mismatches=mismatch,graph_replay=True))
Path('problems/muse_prefill_layer/dev/coalesced_dequant_2048_validation.json').write_text(json.dumps(records,indent=2))
