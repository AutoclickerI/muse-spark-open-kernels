"""Force synthetic tuning and both expert choices with native SM120 stand-ins."""
import sys,json,types
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from torch.utils import cpp_extension
p=Path('problems/muse_prefill_layer/cand_canonical_autotune.py')
m=le.load_module('canonical_autotune_verify',p)
w=ref.generate_weights(ref.WEIGHTS_SEED)
pipe=m.load_module(None)
group=m._grouped_module()
real_pipe_load=m.load_module
real_group_load=m._grouped_module
seen=[]
def target_standin(loader,module,key,*args):
    realcap=torch.cuda.get_device_capability
    realload=cpp_extension.load_inline
    setattr(m,key,None)
    def fake(**kw):
        assert __import__('os').environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
        seen.append(kw['functions'])
        return module
    try:
        torch.cuda.get_device_capability=lambda *a,**kw:(9,0)
        cpp_extension.load_inline=fake
        return loader(*args)
    finally:
        torch.cuda.get_device_capability=realcap
        cpp_extension.load_inline=realload
m.load_module=lambda path:target_standin(real_pipe_load,pipe,'_mod',path)
m._grouped_module=lambda:target_standin(real_group_load,group,'_grouped_mod')
m.IS_HOPPER=True
st=m.prepare(w)
assert seen==[['moe_gemm'],['gemm']],seen
assert st['w13p'].numel()+st['w2p'].numel()==905969664
assert 'w13' not in st and st['bank'] is None
assert set(st['expert_dispatch'])=={753,3012,16384}
print('SM90 prepare targets forced; synthetic cold-L2 tuning with SM120 stand-ins PASS',flush=True)
problem=le.Problem(ref,w)
original=sys.modules.get('sgl_kernel.flash_attn')
stub=types.ModuleType('sgl_kernel.flash_attn');sys.modules['sgl_kernel.flash_attn']=stub
active={}
def fake_attention(**kw):
    data=active['data'];cfg=data['config'];L=cfg['q_len'];B=cfg['batch']
    out=torch.empty_like(kw['q'],dtype=torch.bfloat16)
    m._attn_fwd_kernel[(m.triton.cdiv(L,16),B*m.HKV)](
        kw['q'],kw['k_cache'],kw['v_cache'],kw['page_table'],data['seq_lens'],out,
        data['k_scale'],data['v_scale'],kw['softmax_scale'],L,data['page_table'].shape[1],
        PAGE=cfg['page_size'],WINDOW=cfg['window'],BQ=16,BN=64,num_warps=8,num_stages=2)
    return (out,torch.empty(0,device=out.device)) if active['tuple'] else out
stub.flash_attn_with_kvcache=fake_attention
class Side:
    def call(self,data):
        active['data']=data
        return m.kernel_fn(data,st)
try:
    for choice in ('pipeline','grouped'):
        st['expert_dispatch']={k:choice for k in st['expert_dispatch']}
        for i,layer in enumerate(('full','sliding')):
            active['tuple']=bool(i)
            case=dict(batch=1,ctx=1025,layer=layer,q_len=257,page=64,kv_scale=1/64,spread='agentic')
            data=problem.make_input(case,84401)
            graph,out=le.capture(Side().call,data)
            for seed in (84402,84403):
                fresh=problem.make_input(case,seed);le.copy_into(data,fresh)
                graph.replay();torch.cuda.synchronize()
                ok,msg=problem.check(data,out);assert ok,msg
            print(choice,layer,'forced Hopper FA3 stand-in and fresh graph PASS',flush=True)
finally:
    if original is None:sys.modules.pop('sgl_kernel.flash_attn',None)
    else:sys.modules['sgl_kernel.flash_attn']=original
