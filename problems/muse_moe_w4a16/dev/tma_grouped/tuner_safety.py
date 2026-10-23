"""Force every native choice's Python ABI and reject faults in all tuner phases."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[4]
path=ROOT/'problems/muse_moe_w4a16/cand_tma_grouped.py'
spec=importlib.util.spec_from_file_location('tg_safety',path)
tg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tg;spec.loader.exec_module(tg)


class PortableNative:
    def __init__(self):self.seen=[]
    def tg_capacity(self,rows,width,stages):
        assert rows in (128,256) and width in (32,64) and stages in (3,5)
        return 1
    def tg_run(self,x,ids,tw,pnw,w13,s13,w2,s2,gs13,gs2,inter,y,counter,out,start,width,stages,grid,min_n):
        m=x.shape[0]
        assert ids.shape==(m,8) and tw.shape==(m,8)
        assert ids.dtype==torch.int32 and tw.dtype==torch.float32
        assert pnw.shape==(4096,) and pnw.dtype==torch.bfloat16
        assert inter.shape==(m*8,4096) and inter.dtype==torch.bfloat16
        assert y.shape==(m*8,4096) and y.dtype==torch.float32
        assert out.shape==x.shape and out.dtype==torch.bfloat16
        assert counter.shape==(2,) and counter.dtype==torch.int32
        assert start==64 and 0<grid<=132 and min_n in (8,16)
        assert w13.shape==(32,64,64,2,128,16) and s13.shape==(32,64,64,2,32,4,2)
        assert w2.shape==(32,32,64,2,128,16) and s2.shape==(32,32,64,2,32,4,2)
        self.seen.append((m,width,stages,grid,min_n))
        out.copy_(x)


mod=PortableNative();dev='cuda'
def metadata(shape,dtype):return torch.empty(1,device=dev,dtype=dtype).expand(shape)
bank=dict(mod=mod,pnw=metadata((4096,),torch.bfloat16),
    w13=metadata((32,64,64,2,128,16),torch.uint8),w13_sf=metadata((32,64,64,2,32,4,2),torch.uint8),
    w2=metadata((32,32,64,2,128,16),torch.uint8),w2_sf=metadata((32,32,64,2,32,4,2),torch.uint8),
    gs13=metadata((32,),torch.float32),gs2=metadata((32,),torch.float32),
    counter=torch.zeros(2,device=dev,dtype=torch.int32),sms=132,grids={})
for m in (128,256):
    data=dict(hidden_states=torch.randn(m,4096,device=dev,dtype=torch.bfloat16),
        topk_ids=torch.full((m,8),64,device=dev,dtype=torch.int32),
        topk_weights=torch.ones(m,8,device=dev),config={'local_start':64})
    for c in tg.TG_CHOICES:
        tg._tg_configure(bank,m,c)
        torch.testing.assert_close(tg._tg_run(data,bank,c),data['hidden_states'],rtol=0,atol=0)
assert len(mod.seen)==24
assert 'compute_90a' in ' '.join(tg._tg_cuda_flags((9,0)))
assert 'compute_90a' not in ' '.join(tg._tg_cuda_flags((12,0)))

# Real graph capture and refresh remain in this test. Only expensive arithmetic,
# native dispatch and timers are stand-ins with explicit signatures.
tg._tg_router_weights=lambda w:{}
tg.make_fixture=lambda w,m,seed,mode:dict(hidden_states=torch.ones(m,4096,device=dev,dtype=torch.bfloat16),
    topk_ids=torch.zeros(m,8,device=dev,dtype=torch.int32),topk_weights=torch.ones(m,8,device=dev),config={})
tg._tg_reference=lambda d,w:d['hidden_states'].clone()
tg._tg_incumbent_kernel=lambda d,s:d['hidden_states'].clone()
tg.route_summary=lambda d:dict(pairs=d['topk_ids'].numel(),distinct=1,max_expert=d['topk_ids'].numel())
tg._tg_configure=lambda b,m,c:None
native_run=tg._tg_run
def portable_run(d,b,c):return d['hidden_states'].clone()
def portable_pair(old,new,datasets,flush,repeats=20,deadline=float('inf')):return 100.,90.,.9,repeats
tg._tg_run=portable_run;tg._tg_pair=portable_pair
weights={'w13':torch.empty(1,device=dev)}
cases=[]
for phase in ('configure','capture','oracle','timing','final','tuple'):
    configure=tg._tg_configure;run=tg._tg_run;check=tg._tg_check;pair=tg._tg_pair
    def bad(*a,**k):raise AttributeError('injected missing native entry point')
    if phase=='configure':tg._tg_configure=bad
    if phase=='capture':tg._tg_run=bad
    if phase=='oracle':
        def incorrect(d,b,c):return torch.zeros_like(d['hidden_states'])
        tg._tg_run=incorrect
    if phase=='timing':tg._tg_pair=bad
    if phase=='final':
        def final_bad(old,new,datasets,flush,repeats=20,deadline=float('inf')):
            if repeats==20:raise ValueError('injected finalist')
            return portable_pair(old,new,datasets,flush,repeats,deadline)
        tg._tg_pair=final_bad
    if phase=='tuple':tg._tg_run=lambda d,b,c:(d['hidden_states'],)
    st={'tg_bank':bank,'tg_summary':[],'tg_dispatch':{}}
    tg._tg_tune(weights,st,time.monotonic())
    assert not st['tg_dispatch'],(phase,st['tg_dispatch'])
    assert any('TG_REJECT' in s for s in st['tg_summary']),phase
    cases.append(phase)
    tg._tg_configure=configure;tg._tg_run=run;tg._tg_check=check;tg._tg_pair=pair
st={'tg_bank':bank,'tg_summary':[],'tg_dispatch':{}}
tg._tg_tune(weights,st,time.monotonic()-240)
assert not st['tg_dispatch'] and all('TG_BUDGET' in s for s in st['tg_summary'])
tg._tg_incumbent_prepare=lambda w:{'incumbent_mark':'kept'}
tg._tg_prepare_bank=lambda w:(_ for _ in ()).throw(AttributeError('missing optional native method'))
st=tg.prepare(weights)
assert st['incumbent_mark']=='kept' and not st['tg_dispatch']
assert torch.equal(tg.kernel_fn(data,st),data['hidden_states'])
print('TG_GLUE_SAFETY',json.dumps(dict(choices=len(mod.seen),hopper_flags=True,
    reject_phases=cases,expired_budget=True,wrapper_failure_retains_incumbent=True)),flush=True)
