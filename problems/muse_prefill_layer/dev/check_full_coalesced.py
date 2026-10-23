from pathlib import Path
import sys,json
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from nvfp4 import dequantize
P=Path('problems/muse_prefill_layer');m=le.load_module('full_coalesced_check',P/'cand_expert_portfolio.py')
w=ref.generate_weights(ref.WEIGHTS_SEED);st=m.prepare(w);records=[]
counts=torch.ones(1,device='cuda',dtype=torch.int32)
for gated,rows,bank,key in ((True,8192,st['w13p'][:1],'w13'),(False,4096,st['w2p'][:1],'w2')):
    expected=dequantize(w[key][0],w[key+'_sf'][0],1.).to(torch.bfloat16)
    for kb in (1024,2048):
        out=torch.empty((1,rows,4096),device='cuda',dtype=torch.bfloat16)
        m._canonical_expand_coalesced[(1,rows//16,4096//kb)](bank,bank.view(torch.int32),out.view(torch.int32),counts,ROWS=rows,GATED=gated,KB=kb,num_warps=8)
        mismatch=int((out[0]!=expected).sum());assert mismatch==0
        row=dict(weight=key,kb=kb,values=expected.numel(),mismatches=mismatch)
        print(row,flush=True);records.append(row)
(P/'dev/full_coalesced_validation.json').write_text(json.dumps(records,indent=2))
