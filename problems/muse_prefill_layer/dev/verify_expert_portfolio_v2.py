"""Exercise all portfolio math, force SM90 prepare and capture fresh layer inputs."""
from pathlib import Path
import sys,json,types,time
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from torch.utils import cpp_extension
P=Path('problems/muse_prefill_layer')
target=Path(sys.argv[1]) if len(sys.argv)>1 else P/'cand_expert_portfolio_v2.py'
m=le.load_module('expert_portfolio_verify',target)
w=ref.generate_weights(ref.WEIGHTS_SEED)
actual=m._grouped_module();real=m._grouped_module
seen=[]
def force_constructor():
    oldcap=torch.cuda.get_device_capability;oldload=cpp_extension.load_inline
    m._mod=None
    def fake(**kw):
        assert __import__('os').environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
        seen.append(kw['functions']);return actual
    try:
        torch.cuda.get_device_capability=lambda *a,**kw:(9,0)
        cpp_extension.load_inline=fake
        return real()
    finally:
        torch.cuda.get_device_capability=oldcap;cpp_extension.load_inline=oldload
m._grouped_module=force_constructor
m.IS_HOPPER=True
t0=time.time();st=m.prepare(w)
assert seen==[['gemm','moe_gemm']]
assert st['w13p'].numel()+st['w2p'].numel()==905969664
assert st['bank'] is None and 'w13' not in st and 'w2' not in st
print('Forced SM90 constructor, all synthetic grouped/DG choices PASS, prepare_s',time.time()-t0,flush=True)
problem=le.Problem(ref,w)
original=sys.modules.get('sgl_kernel.flash_attn');stub=types.ModuleType('sgl_kernel.flash_attn');sys.modules['sgl_kernel.flash_attn']=stub
active={}
def fake_fa3(**kw):
    d=active['data'];cfg=d['config'];L=cfg['q_len'];B=cfg['batch']
    out=torch.empty_like(kw['q'],dtype=torch.bfloat16)
    m._attn_fwd_kernel[(m.triton.cdiv(L,16),B*m.HKV)](kw['q'],kw['k_cache'],kw['v_cache'],kw['page_table'],d['seq_lens'],out,
      d['k_scale'],d['v_scale'],kw['softmax_scale'],L,d['page_table'].shape[1],PAGE=cfg['page_size'],WINDOW=cfg['window'],BQ=16,BN=64,num_warps=8,num_stages=2)
    return (out,torch.empty(0,device=out.device)) if active['tuple'] else out
stub.flash_attn_with_kvcache=fake_fa3
class Side:
    def call(self,d):active['data']=d;return m.kernel_fn(d,st)
records=[]
try:
    for name in (('pipeline','dg_hint','dg_hint_k2048') if '--selected-only' in sys.argv else tuple(m._GROUPED_CHOICES)+('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048','dg_w8','dg_hint_w8','dg_k2048_w8','dg_hint_k2048_w8')):
        st['expert_dispatch']={753:name,3012:name,16384:name}
        for i,layer in enumerate(('full','sliding')):
            active['tuple']=bool(i)
            case=dict(batch=1,ctx=1025,layer=layer,q_len=139,page=64,kv_scale=1/64,spread='agentic')
            d=problem.make_input(case,8921)
            graph,out=le.capture(Side().call,d)
            for seed in (8922,8923):
                fresh=problem.make_input(case,seed);le.copy_into(d,fresh)
                graph.replay();torch.cuda.synchronize()
                ok,msg=problem.check(d,out);assert ok,(name,layer,msg)
                records.append(dict(choice=name,layer=layer,seed=seed,ok=ok,msg=msg))
            print(name,layer,'fresh graph FP32 layer reference PASS',flush=True)
            del graph,out,d,fresh
finally:
    if original is None:sys.modules.pop('sgl_kernel.flash_attn',None)
    else:sys.modules['sgl_kernel.flash_attn']=original
(P/'dev'/(target.stem+'_validation.json')).write_text(json.dumps(records,indent=2))
