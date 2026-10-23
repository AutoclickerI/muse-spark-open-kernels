"""Exercise actual packed encoding and GEMM with every finite E4M3 scale and E2M1 code."""
import sys,torch,triton
sys.path[:0]=['problems/muse_prefill_layer']
import cand_v17 as m
from nvfp4 import dequantize
T,K,N=129,4096,256
# Every signed finite E4M3 byte, including subnormals and zero; remove NaNs.
b=torch.arange(N*(K//16),device='cuda',dtype=torch.int32).reshape(1,N,K//16)%256
b=torch.where((b&127)==127,0,b).to(torch.uint8)
sf=b.view(torch.float8_e4m3fn)
# Cover every pair of FP4 codes across positions, including negative zero.
w=(torch.arange(N*(K//2),device='cuda',dtype=torch.int32).reshape(1,N,K//2)%256).to(torch.uint8)
wb,sb=m.repack_fast(w,sf,False)
x=torch.randn(T,K,device='cuda',dtype=torch.bfloat16)
es=torch.full((T,8),-1,device='cuda',dtype=torch.int32);es[:,0]=0
pos=torch.zeros_like(es);pos[:,0]=torch.arange(T,device='cuda',dtype=torch.int32)
cnt=torch.zeros(32,device='cuda',dtype=torch.int32);cnt[0]=T
ts=torch.empty(33,device='cuda',dtype=torch.int32)
m.tile_offsets_kernel[(1,)](cnt,ts,BNC=128)
xt=torch.empty(2*128*K,device='cuda',dtype=torch.bfloat16)
rows=torch.empty_like(es)
m.gather_tiled_kernel[(T,)](x,es,pos,ts,xt,rows,D=K,BNC=128,num_warps=4)
yt=torch.empty(2*128*N,device='cuda',dtype=torch.bfloat16)
gs=torch.ones(1,device='cuda',dtype=torch.float32)
mod=m.load_module(None)
mod.moe_gemm(wb,sb,gs,xt,yt,ts,1,64,4,0,132)
out=torch.empty(T,N,device='cuda',dtype=torch.bfloat16)
m.untile_rows_kernel[(T,)](yt,rows[:,0].contiguous(),out,D=N,BNC=128)
ref=x.float()@dequantize(w[0],sf[0],gs[0]).T
err=float((out.float()-ref).norm()/ref.norm())
print('Every finite E4M3 scale, all FP4 codes: relative L2',err, 'finite',bool(out.isfinite().all()),flush=True)
assert err<0.003 and bool(out.isfinite().all())
