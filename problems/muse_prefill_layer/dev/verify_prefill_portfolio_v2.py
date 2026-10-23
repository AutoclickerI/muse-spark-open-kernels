"""Force both SM90 constructors and synthetic tuners through native sm120 stand-ins."""
import sys,time,types
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer']
import torch
import local_eval as le
import reference as ref
from torch.utils import cpp_extension
p=Path('problems/muse_prefill_layer/cand_prefill_portfolio_v2.py')
m=le.load_module('prefill_autotune_verify',p)
w=ref.generate_weights(ref.WEIGHTS_SEED)
group=m._grouped_module();real_group_load=m._grouped_module
seen=[]
def force_group():
    cap=torch.cuda.get_device_capability;inline=cpp_extension.load_inline
    m._mod=None
    def fake(**kw):
        assert __import__('os').environ['TORCH_CUDA_ARCH_LIST']=='9.0a'
        seen.append(kw['functions']);return group
    try:
        torch.cuda.get_device_capability=lambda *a,**kw:(9,0)
        cpp_extension.load_inline=fake
        return real_group_load()
    finally:
        torch.cuda.get_device_capability=cap;cpp_extension.load_inline=inline
m._grouped_module=force_group
original=sys.modules.get('sgl_kernel.flash_attn')
stub=types.ModuleType('sgl_kernel.flash_attn');sys.modules['sgl_kernel.flash_attn']=stub
as_tuple=False

def fake_attention(q,k_cache,v_cache,cache_seqlens,cu_seqlens_q,max_seqlen_q,
                  softmax_scale,causal,window_size,page_table=None,k_descale=None,v_descale=None,**kw):
    assert causal
    B=cu_seqlens_q.numel()-1;L=max_seqlen_q;T=B*L
    seq=cache_seqlens-L
    one=torch.ones(1,device=q.device)
    ks=one if k_descale is None else k_descale.reshape(-1)[:1]
    vs=one if v_descale is None else v_descale.reshape(-1)[:1]
    out=torch.empty((T,1024),device=q.device,dtype=torch.bfloat16)
    window=window_size[0]+1 if window_size[0]>=0 else 0
    m._attn_fwd_kernel[(m.triton.cdiv(L,16),B*2)](q,k_cache,v_cache,page_table,seq,out,ks,vs,
      softmax_scale,L,page_table.shape[1],PAGE=k_cache.shape[1],WINDOW=window,BQ=16,BN=64,
      num_warps=8,num_stages=2)
    return (out.view(T,16,64),None) if as_tuple else out.view(T,16,64)
stub.flash_attn_with_kvcache=fake_attention
m.IS_HOPPER=True
try:
    t0=time.perf_counter();st=m.prepare(w);elapsed=time.perf_counter()-t0
    assert elapsed<300,elapsed
    assert seen==[['gemm','moe_gemm']],seen
    assert st['w13p'].numel()+st['w2p'].numel()==905969664
    assert 'w13' not in st and st['bank'] is None
    assert set(st['expert_dispatch'])=={753,3012,16384}
    assert set(st['attention_dispatch'])=={0,32768,65536}
    print('Forced SM90 prepare and both synthetic tuners PASS',elapsed,st['expert_dispatch'],st['attention_dispatch'],flush=True)
    problem=le.Problem(ref,w)
    class Side:
        def call(self,d):return m.kernel_fn(d,st)
    tests=le.tests_of(Path('problems/muse_prefill_layer/TASK.md').read_text())
    results=le.run_tests(problem,Side(),tests,28931)
    assert all(r['ok'] for r in results),results
    # Full-layer production dispatch with unequal/nearby context, page64 and non-unit scales.
    # Force each complete attention choice independently after checking the tuned selection.
    for i,name in enumerate(('selected','fa3','fp8_direct128','fp8_gather128','fp8_gather256')):
        as_tuple=bool(i%2)
        if name!='selected':st['attention_dispatch'][32768]=name
        case=dict(batch=1,ctx=32768,layer='full',q_len=2048,page=64,kv_scale=1/64,spread='agentic')
        d=problem.make_input(case,44201)
        graph,out=le.capture(Side().call,d)
        for seed in (44202,44203):
            fresh=problem.make_input(case,seed);le.copy_into(d,fresh)
            graph.replay();torch.cuda.synchronize()
            ok,msg=problem.check(d,out);assert ok,msg
        print('Full prefill autotune',name,'forced Hopper stand-in, actual expert+attention production fresh graphs PASS',flush=True)
        del graph,out,d,fresh
finally:
    if original is None:sys.modules.pop('sgl_kernel.flash_attn',None)
    else:sys.modules['sgl_kernel.flash_attn']=original
