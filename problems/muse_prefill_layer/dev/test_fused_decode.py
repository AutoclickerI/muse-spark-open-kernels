"""Exhaustive represented-value test of the actual packed decoder used by direct GEMM."""
from pathlib import Path
import sys,json,hashlib
sys.path[:0]=['tools']
import torch,triton
import triton.language as tl
import local_eval as le
p=Path('problems/muse_prefill_layer/cand_fused_expert_portfolio.py');m=le.load_module('fused_decode_test',p)
@triton.jit
def probe(Q,SF,OUT,N:tl.constexpr,RESTORE:tl.constexpr,B:tl.constexpr):
 i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
 q=tl.load(Q+i,ok,0).to(tl.int32)
 sf=tl.load(SF+i,ok,0).to(tl.float8e4nv,bitcast=True).to(tl.float32)
 a,b=m._fused_decode_pair(q,sf,RESTORE_256=RESTORE)
 if not RESTORE:a=(a.to(tl.float32)*256).to(tl.bfloat16);b=(b.to(tl.float32)*256).to(tl.bfloat16)
 tl.store(OUT+2*i,a,ok);tl.store(OUT+2*i+1,b,ok)
sf_codes=torch.tensor([s for s in range(256) if s not in (127,255)],device='cuda',dtype=torch.uint8)
q=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8).repeat(sf_codes.numel()).contiguous()
s=sf_codes.repeat_interleave(256).contiguous();n=q.numel();out=torch.empty(n*2,device='cuda',dtype=torch.bfloat16)
lookup=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
expected=torch.stack((lookup[(q&15).long()],lookup[(q>>4).long()]),dim=1)*s.view(torch.float8_e4m3fn).float()[:,None]
expected=expected.reshape(-1).bfloat16()
for restore in (True,False):
 for _ in range(2):probe[(triton.cdiv(n,256),)](q,s,out,N=n,RESTORE=restore,B=256)
 torch.cuda.synchronize();assert torch.equal(out,expected),int(torch.count_nonzero(out!=expected))
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):probe[(triton.cdiv(n,256),)](q,s,out,N=n,RESTORE=restore,B=256)
 g.replay();torch.cuda.synchronize();assert torch.equal(out,expected)
 print('Packed BF16 decode exact represented values PASS',restore,n*2,flush=True)
p.with_name('dev').joinpath('fused_decode_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),hardware='sm120',weights_per_variant=n*2,variants=2,all_finite_signed_sf=True,exact=True),indent=2))
