"""Isolate the old last-warp combine and compare metadata-hoisted/vector CTA tails."""
import sys,json,hashlib,statistics
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'prof/lab'))
import moe_fused_m256 as lab

def main():
 s=(H/'tail_probe.cu').read_text();target='-gencode=arch=compute_120,code=sm_120'
 mod=load_inline(name='tail_probe_'+hashlib.sha256(s.encode()).hexdigest()[:12],cpp_sources='void tp_run(std::vector<torch::Tensor>,int64_t,int64_t,int64_t);',cuda_sources=s,functions=['tp_run'],extra_cuda_cflags=['-O3','-lineinfo',target],verbose=False)
 w=lab.dc_weights(5353,'cuda');ds=lab.dc_fixtures(w,256,True);ids=ds[0]['topk_ids'].int().contiguous();tw=ds[0]['topk_weights'].float();rows=[]
 for nt in (16,32):
  count=torch.bincount(ids[(ids>=0)&(ids<32)],minlength=32).int();off=torch.cat((count.new_zeros(1),((count+nt-1)//nt*nt).cumsum(0))).int();cap=int(off[-1]);order=count.argsort(descending=True).int();task=torch.cat((count.new_zeros(1),((count[order.long()]+nt-1)//nt).cumsum(0))).int()
  slots=torch.full((32,256),-1,device='cuda',dtype=torch.int32);pos=torch.full_like(ids,-1)
  for e in range(32):
   ii=(ids.flatten()==e).nonzero().flatten();slots[e,:ii.numel()]=ii.int();pos.flatten()[ii]=torch.arange(ii.numel(),device='cuda',dtype=torch.int32)
  y=torch.randn(cap,4096,device='cuda');ss=y.square().reshape(cap,64,64).sum(-1);inv=torch.empty(cap,device='cuda');ready=torch.empty(256,device='cuda',dtype=torch.int32);n=((ids>=0)&(ids<32)).sum(-1).int();out=torch.empty(256,4096,device='cuda',dtype=torch.bfloat16)
  v=[count,off,order,task,slots,ids,tw,pos,y,ss,inv,ready,n,out];ref=torch.zeros_like(out,dtype=torch.float32)
  for j in range(8):
   ok=(ids[:,j]>=0)&(ids[:,j]<32);p=off[ids[ok,j].long()]+pos[ok,j];ref[ok]+=y[p.long()]*torch.rsqrt(y[p.long()].square().mean(-1,keepdim=True)+1e-8)*tw[ok,j,None]
  flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8)
  graphs=[]
  for mode in (0,1,2,3):
   ready.zero_();out.zero_();mod.tp_run(v,mode,nt,528);torch.cuda.synchronize();err=float((out.float()-ref).norm()/ref.norm());assert mode==3 or err<.01,(nt,mode,err)
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):mod.tp_run(v,mode,nt,528)
   graphs.append(graph)
  times=[[] for _ in graphs];a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
  for rep in range(12):
   for mode in ((0,1,2,3) if rep%2==0 else (3,2,1,0)):
    ready.zero_();out.zero_();flush.zero_();a.record();graphs[mode].replay();b.record();b.synchronize();times[mode].append(a.elapsed_time(b)*1000)
  row=dict(nt=nt,groups=int(task[-1]),pairs=int(count.sum()),experts=int((count>0).sum()),old_us=statistics.median(times[0]),hoist_us=statistics.median(times[1]),cta_us=statistics.median(times[2]),norm_tickets_us=statistics.median(times[3]));rows.append(row);print(row,flush=True)
 (H/'diagnostics/tail_probe.json').write_text(json.dumps(dict(hardware=torch.cuda.get_device_name(),rows=rows),indent=2)+'\n')

if __name__=='__main__':main()
