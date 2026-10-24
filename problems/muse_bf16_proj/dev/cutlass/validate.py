"""Force every Hopper CUTLASS Python/ABI branch on GPU0 with strict portable math.

These are glue/capture checks on sm_120, not executions or timings of Hopper SASS.
The real 16 collective specializations are independently compiled for sm_90a.
"""
from pathlib import Path
import importlib.util
import time
from types import SimpleNamespace
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'prof/lab/verify_proj_cutlass.py'
spec=importlib.util.spec_from_file_location('vgc_lab',P)
lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
spec=importlib.util.spec_from_file_location('vgc_install',P.parent.parent.parent/'problems/muse_bf16_proj/dev/cutlass/projection_cutlass.py')
block=importlib.util.module_from_spec(spec);spec.loader.exec_module(block)

class AbiStandin:
    def __init__(self):
        self.handles={};self.tensors={};self.next=1;self.seen=set();self.calls=0
        self.error=b'injected Hopper-only attribute/launch failure'
        self.inject=None
    def vgc_error(self):return self.error
    def vgc_create(self,tile,fp32,m,n,k,sk,mode,sms,raster,sw,red):
        assert 0<=tile<8 and fp32 in (0,1) and mode in (0,1,2,3)
        assert sk>=1 and (sk==1 or mode in (0,2)) and sms<=132
        self.seen.add((m,n,k,fp32,tile,mode,sk,sms,raster,sw,red))
        p=self.next;self.next+=1
        self.handles[p]=(tile,fp32,m,n,k)
        return p
    def vgc_workspace(self,p):assert p in self.handles;return 256
    def vgc_info(self,p,index):assert p in self.handles and index in (0,1,2);return (3,384,200000)[index]
    def vgc_destroy(self,p):assert p in self.handles;self.handles.pop(p)
    def vgc_run(self,p,x,w,y,ws,stream):
        tile,fp32,m,n,k=self.handles[p]
        a,b,c=self.tensors[x],self.tensors[w],self.tensors[y]
        assert a.shape==(m,k) and b.shape==(n,k) and c.shape==(m,n)
        assert c.dtype==(torch.float32 if fp32 else torch.bfloat16)
        assert ws and stream==torch.cuda.current_stream().cuda_stream
        self.calls+=1
        if self.inject=='attribute':
            self.inject=None;raise AttributeError('strict missing entry point')
        if self.inject=='runtime':
            self.inject=None;return -1
        torch.mm(a,b.T,out=c,**({'out_dtype':torch.float32} if fp32 else {}))
        if self.inject=='wrong':
            c.zero_()
        return 0

abi=AbiStandin()
class Plan(lab.VgcPlan):
    def run(self,x,w,out):
        abi.tensors.update({t.data_ptr():t for t in (x,w,out)})
        return super().run(x,w,out)
class Module:
    lib=abi
    def error(self):return RuntimeError(abi.vgc_error().decode())
    def plan(self,m,n,k,dtype,recipe,device):return Plan(self,m,n,k,dtype,recipe,device)

class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)
fake_library=SimpleNamespace(**{name:Function(getattr(abi,name)) for name in
    ('vgc_error','vgc_create','vgc_workspace','vgc_info','vgc_destroy','vgc_run')})
def forced_compile(**kwargs):
    assert kwargs['cuda_sources']==lab.VGC_SOURCE
    assert kwargs['extra_include_paths']==[
        '/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include']
    assert '-DNDEBUG' in kwargs['extra_cuda_cflags']
    assert lab._vgos.environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
    return '/portable/strict_abi.so'
lab.VgcPlan=Plan
with patch('torch.cuda.get_device_capability',return_value=(9,0)), \
        patch('torch.utils.cpp_extension.load_inline',side_effect=forced_compile), \
        patch.object(lab._vgct,'CDLL',return_value=fake_library):
    module=lab.vgc_module()
    assert module is lab.vgc_module()
assert len(fake_library.vgc_create.argtypes)==11 and len(fake_library.vgc_run.argtypes)==6
weights={name:torch.randn(n,k,device='cuda',dtype=dtype) for name,n,k,dtype in (
    ('qkv_gate',2304,8192,torch.bfloat16),('o',8192,1024,torch.bfloat16),
    ('pooler',4096,8192,torch.bfloat16),('unpooler',8192,4096,torch.bfloat16),
    ('router',256,8192,torch.float32))}
lab.vgc_module=lambda:module
# The H200 has 132 SMs; force that hardware selection without mocking input math.
original_choices=lab.vgc_choices
lab.vgc_choices=lambda m,n,k,sms:original_choices(m,n,k,132)
counts={'pairs':0,'twenty':0}
def checked_pair(old,new,x,banks,flush,repeats=20):
    assert repeats in (4,20) and len(banks)==3
    counts['pairs']+=1;counts['twenty']+=int(repeats==20)
    for fresh in banks:
        x.copy_(fresh);old.graph.replay();new.graph.replay();lab.vp_check(new.out,old.out)
    # Deterministic ordering tests selection. These mock numbers are NEVER performance evidence.
    return 10.,9.,.9
lab.vp_pair=checked_pair
abi.inject='attribute'
rows=lab.lab(time.monotonic()+225,weights)
assert sum(r.startswith('VGC_BEST ') for r in rows)==8,rows
assert any('AttributeError' in r for r in rows),rows
assert all(any(s[4]==tile and s[5]==mode for s in abi.seen) for tile in range(8) for mode in range(4))
assert all(any(s[0]==m and s[3]==fp32 and s[4]==tile for s in abi.seen)
           for m in (128,256) for fp32 in (0,1) for tile in range(8))
assert counts['twenty']>=24,counts
assert not abi.handles
print('FORCED_LAB sm120 families',len(abi.seen),'ABI_calls',abi.calls,'paired20',counts['twenty'],
      'log_chars',sum(len(r)+1 for r in rows))
assert sum(len(r)+1 for r in rows)<9000
for r in rows:
    if r.startswith(('VGC_BEST','VGC_REJECT','VGC_END')):print(r)

block.vgc_module=lambda:module
block.vp_pair=checked_pair
hi=weights['router'].bfloat16();lo=(weights['router']-hi.float()).bfloat16()
state={**weights,'pool_router':torch.cat((weights['pooler'],hi,lo),0)}
def incumbent(x,w,name,state,out_dtype=torch.bfloat16):
    y=torch.mm(x,w.T,out_dtype=torch.float32) if out_dtype==torch.float32 else x@w.T
    return y.unsqueeze(0),1
recipes={(m,name):[(tile,mode,sk,132,0,1,red) for tile,mode,sk,red in
                   ((0,3,1,0),(6,2,4,1),(7,1,1,0))]
         for m in (128,256) for name in ('qkv_gate','pool_router','unpooler','o')}
notes=block.vgc_prepare(state,incumbent,time.monotonic()+240,recipes)
assert len(state['_vgc_proj']['plans'])==8,notes
bound=block.vgc_bind(incumbent)
checks=0
for (m,name),saved in state['_vgc_proj']['plans'].items():
    w=state[name];dtype=torch.float32 if name=='pool_router' else torch.bfloat16
    x=torch.randn(m,w.shape[1],device='cuda',dtype=torch.bfloat16)
    with lab._PgGraph(lambda:block._vgc_materialize(bound(x,w,name,state,out_dtype=dtype))) as graph:
        for _ in range(3):
            x.normal_();graph.graph.replay();lab.vp_check(graph.out,x.float()@w.float().T);checks+=1
    # Guard changes in strides, weight identity and output dtype.
    for xx,ww,dt in ((x,w.clone(),dtype),(x[::2],w,dtype),(x,w,torch.float32)):
        got=block._vgc_materialize(bound(xx,ww,name,state,out_dtype=dt))
        lab.vp_check(got,xx.float()@ww.float().T);checks+=1
for plan,*_ in state['_vgc_proj']['plans'].values():plan.close()
for inject in ('runtime','wrong','attribute'):
    abi.inject=inject
    notes=block.vgc_prepare(state,incumbent,time.monotonic()+240,{(128,'qkv_gate'):[(0,3,1,132,0,1,0)]})
    assert not state['_vgc_proj']['plans'] and any('REJECT' in r for r in notes),(inject,notes)
    abi.inject=None
notes=block.vgc_prepare(state,incumbent,time.monotonic()-1,recipes)
assert 'BUDGET' in notes[0] and not state['_vgc_proj']['plans']
notes=block.vgc_prepare(state,incumbent,time.monotonic()+240,{(128,'qkv_gate'):[(100,3,1,132,0,1,0)]})
assert not state['_vgc_proj']['plans'] and any('REJECT' in r for r in notes)
assert not abi.handles
assert lab._PgGraph.active==0
print('INSTALL sm120 checks',checks,'rejection phases runtime/accuracy/attribute/budget/unknown_tile PASS')
