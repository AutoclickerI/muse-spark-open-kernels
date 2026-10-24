"""Emulate Hopper's absolute B128 swizzle on the local MMA path."""
from pathlib import Path
import ast
import hashlib
import json
import torch
from torch.utils.cpp_extension import load_inline
from vector import vec_prepare,vec_work,VECTOR_CPP
from validate import make_weights,load_helpers

HERE=Path(__file__).resolve().parent


def main():
    w=make_weights();base=vec_prepare(w);ref=load_helpers()
    # Physical SMEM address participates in the Hopper swizzle. The original
    # local reader used only row-relative addresses, hiding a missing phase.
    tree=ast.parse((HERE.parents[3]/'prof/lab/moe_vector_wgmma_seed.py').read_text())
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='VECTOR_CUDA' for t in n.targets))
    a=src.index('#else',src.index('template<int NT>__device__ __forceinline__ void mma'))
    b=src.index('template<int NT>__device__ __forceinline__ void load_stage',a)
    raw=src[:a]+src[a:b].replace('((r&7)*8)','(((r+(smaddr(xs)>>7))&7)*8)')+src[b:]
    aligned=raw.replace('extern __shared__ __align__(128) unsigned char sm[];',
       'extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);')
    aligned=aligned.replace('auto*bs=(__nv_bfloat16*)(ss+2*SB);',
                            'auto*bs=(__nv_bfloat16*)(sm+((2*QB+2*SB+1023)&~1023));')
    aligned=aligned.replace('2*mwvec::QB+2*mwvec::SB+2*NT*mwvec::BK*2',
                    '1023+((2*mwvec::QB+2*mwvec::SB+1023)&~1023)+2*NT*mwvec::BK*2')
    result={};m=256;torch.manual_seed(53)
    ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int();ids[:,0]=0
    ids[:,1:]=torch.where(ids[:,1:]==0,255,ids[:,1:])
    data=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
              topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0})
    want=ref(data,w)
    for label,code in (('original_absolute_swizzle',raw),('aligned_absolute_swizzle',aligned)):
        mod=load_inline(name='mw_desc_'+hashlib.sha256(code.encode()).hexdigest()[:12],cpp_sources=VECTOR_CPP,
              cuda_sources=code,functions=['vec_run'],extra_cuda_cflags=['-O3','-std=c++20'],verbose=False)
        bank=dict(base,mod=mod);errs=[]
        for nt in (16,32,64):
            fs=vec_work(data,bank,nt)
            for f in fs:got=f()
            torch.cuda.synchronize();err=float((got.float()-want.float()).norm()/want.float().norm())
            errs.append(err);print(label,nt,err,flush=True)
        result[label]=errs
    assert all(e>1.3 for e in result['original_absolute_swizzle']),result
    assert all(e<.01 for e in result['aligned_absolute_swizzle']),result
    (HERE/'diagnostics/descriptor_agreement.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
