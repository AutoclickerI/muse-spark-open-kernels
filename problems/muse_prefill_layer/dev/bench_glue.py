import sys,torch,triton
sys.path.insert(0,'problems/muse_prefill_layer')
import cand_v1 as v

def bench(fn):
 for _ in range(2): fn()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):
  for _ in range(5): fn()
 s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
 times=[]
 for _ in range(3):
  s.record();g.replay();e.record();e.synchronize();times.append(s.elapsed_time(e)*200)
 return sorted(times)[1]
for T in (753,3012,16384):
 a=torch.randn((T,8192),device='cuda',dtype=torch.bfloat16); s=torch.randn((T,8192),device='cuda'); sn=torch.empty_like(s);o=torch.empty_like(a)
 w=torch.randn((8192,),device='cuda')*.1+1;alpha=torch.sigmoid(w);beta=1-alpha
 for nw in (4,8,16):
  f=lambda:v._boundary_kernel[(T,)](a,s,sn,o,w,alpha,beta,w,HAS_POST_W=True,H=8192,num_warps=nw)
  print('boundary',T,nw,bench(f),flush=True)
 del a,s,sn,o
for T in (753,3012,16384):
 h=torch.randn((T+128,8192),device='cuda',dtype=torch.bfloat16);y=torch.empty((T+128,4096),device='cuda',dtype=torch.bfloat16)
 cnt=torch.zeros(32,device='cuda',dtype=torch.int32);cnt[0]=T
 starts=torch.zeros_like(cnt);gs=torch.ones(32,device='cuda')*.001
 for br in (1,2,4,8,16):
  for nw in (4,8):
   f=lambda:v._swiglu_kernel[(32,triton.cdiv(T,br))](h,y,starts,cnt,gs,BR=br,D=4096,num_warps=nw)
   print('swiglu',T,br,nw,bench(f),flush=True)
