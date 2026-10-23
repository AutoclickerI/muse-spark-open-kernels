"""Dev-only SM90 prepare/FA3 dispatch stands in for kernels unavailable on sm120."""
import sys,types
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from torch.utils import cpp_extension
path=Path(sys.argv[1]);m=le.load_module('canonical_hopper_glue',path)
w=ref.generate_weights(ref.WEIGHTS_SEED)
actual_mod=m._grouped_module()
original_inline=cpp_extension.load_inline
original_cap=torch.cuda.get_device_capability
seen=[]
def fake_inline(**kw):
    seen.append(kw)
    assert kw['functions']==['gemm']
    assert __import__('os').environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
    return actual_mod
try:
    cpp_extension.load_inline=fake_inline
    torch.cuda.get_device_capability=lambda *a,**k:(9,0)
    m._mod=None
    st=m.prepare(w)
finally:
    cpp_extension.load_inline=original_inline
    torch.cuda.get_device_capability=original_cap
assert len(seen)==1
assert 'blob13' not in st and 'w13' not in st and st['bank'] is None
assert st['w13p'].numel()+st['w2p'].numel()==905969664
print('SM90 prepare compiler target forced with native stand-in; one V6 bank PASS',flush=True)
problem=le.Problem(ref,w)
original_fa3=sys.modules.get('sgl_kernel.flash_attn')
stub=types.ModuleType('sgl_kernel.flash_attn')
sys.modules['sgl_kernel.flash_attn']=stub
m.IS_HOPPER=True
active={}
def fake_fa3(**kw):
    d=active['data'];cfg=d['config'];T=cfg['tokens'];L=cfg['q_len'];B=cfg['batch']
    out=torch.empty_like(kw['q'],dtype=torch.bfloat16)
    m._attn_fwd_kernel[(m.triton.cdiv(L,16),B*m.HKV)](
        kw['q'],kw['k_cache'],kw['v_cache'],kw['page_table'],d['seq_lens'],out,
        d['k_scale'],d['v_scale'],kw['softmax_scale'],L,d['page_table'].shape[1],
        PAGE=cfg['page_size'],WINDOW=cfg['window'],BQ=16,BN=64,num_warps=8,num_stages=2)
    return (out,torch.empty(0,device=out.device)) if active['tuple'] else out
stub.flash_attn_with_kvcache=fake_fa3
class Side:
    def call(self,d):
        active['data']=d
        return m.kernel_fn(d,st)
try:
    for layer in ('full','sliding'):
        case=dict(batch=1,ctx=1025,layer=layer,q_len=139,page=64,kv_scale=1/64,spread='agentic')
        for as_tuple in (False,True):
            active['tuple']=as_tuple
            d=problem.make_input(case,64201)
            g,out=le.capture(Side().call,d)
            for seed in (64202,64203):
                fresh=problem.make_input(case,seed);le.copy_into(d,fresh)
                g.replay();torch.cuda.synchronize()
                ok,msg=problem.check(d,out);assert ok,msg
            print('SM90 FA3 glue',layer,'tuple',as_tuple,'fresh graph PASS',flush=True)
finally:
    if original_fa3 is None:sys.modules.pop('sgl_kernel.flash_attn',None)
    else:sys.modules['sgl_kernel.flash_attn']=original_fa3
