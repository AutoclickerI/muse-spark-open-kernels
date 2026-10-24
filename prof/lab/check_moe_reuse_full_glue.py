"""Force the Hopper full-shared Python launch path with a portable native stand-in."""
import importlib.util
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('reuse',root/'problems/muse_moe_w4a16/dev/v9_reuse/reuse.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
real=r.reuse_module()
class StandIn:
    def reuse_route(self,*a):
        assert a[-1]==32
        real.reuse_route(*a)
    def reuse_gemm(self,*a):
        assert a[-3:]==(32,2,self.grid)
        assert a[0].dtype==torch.uint8 and a[1].dtype==torch.float32
        assert a[2].dtype==torch.bfloat16 and a[3].dtype==torch.bfloat16 and a[4].dtype==torch.float32
        if a[-4]==0:a[3].zero_()
        else:a[4].zero_()
        if a[10].numel():a[10].zero_()
mod=StandIn();sms=torch.cuda.get_device_properties(0).multi_processor_count
bank=dict(w13p=torch.empty(1,device='cuda',dtype=torch.uint8),w2p=torch.empty(1,device='cuda',dtype=torch.uint8),
          gs13=torch.ones(32,device='cuda'),gs2=torch.ones(32,device='cuda'),pnw=torch.zeros(4096,device='cuda',dtype=torch.bfloat16),grid=sms,reuse_mod=mod)
for m in (128,256):
    for mult in (2,4):
        mod.grid=sms*mult
        data=dict(hidden_states=torch.randn(m,4096,device='cuda').bfloat16(),
                  topk_ids=torch.randint(-1,256,(m,8),device='cuda',dtype=torch.int32),
                  topk_weights=torch.full((m,8),.125,device='cuda'),config=dict(local_start=0))
        phases,trace=r.reuse_work(data,bank,(32,2,mult),True)
        for f in phases:out=f()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for f in phases:out=f()
        for _ in range(3):
            data['hidden_states'].normal_();data['topk_ids'].random_(-1,256)
            graph.replay();torch.cuda.synchronize()
            assert bool((out==0).all()) and out.dtype==torch.bfloat16
        assert all(s.shape==(mod.grid,12) for s in trace)
print('PASS full-shared Python glue: 4 captured launch paths, 12 fresh replays; full-shared CUDA arithmetic remains H200-only')
