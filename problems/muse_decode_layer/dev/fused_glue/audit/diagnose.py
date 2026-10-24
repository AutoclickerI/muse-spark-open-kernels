"""Local causal ablations for the rejected H200 glue implementation; no sends."""
from pathlib import Path
import hashlib,json,sys,ctypes,time
import torch
from torch.utils.cpp_extension import load_inline
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(HERE.parent)]
import local_eval
from validate import synthetic


def measure(fn,flush,repeats=6):
    for _ in range(2):fn()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);values=[]
    for _ in range(repeats):
        flush.zero_();a.record();g.replay();b.record();b.synchronize();values.append(a.elapsed_time(b)*1000)
    return sorted(values)[len(values)//2]


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    original=(HERE.parent/'kernel.cu').read_text()
    # These variants deliberately delete work. They are not candidate kernels.
    source=original.replace('if(POOL)prologue<T,W>(a,sx);', '''if(POOL){
      if(recipe_flags & 1){for(int i=threadIdx.x;i<T*(a.ks+8);i+=W*32)sx[i]=__float2bfloat16(0.f);__syncthreads();}
      else prologue<T,W>(a,sx);
    }''')
    source=source.replace('void kernel(Args a,bool pdl){','void kernel(Args a,bool pdl,int recipe_flags){')
    source=source.replace('if(MODE==1||MODE==2)route<T,W>(a);else boundary2<T,W>(a);','if(!(recipe_flags & 2)){if(MODE==1||MODE==2)route<T,W>(a);else boundary2<T,W>(a);}')
    source=source.replace('int run(Args a,bool pdl,cudaStream_t stream){','int run(Args a,bool pdl,int recipe_flags,cudaStream_t stream){')
    source=source.replace('cudaLaunchKernelEx(&c,f,a,pdl)','cudaLaunchKernelEx(&c,f,a,pdl,recipe_flags)').replace('>>>(a,pdl);','>>>(a,pdl,recipe_flags);')
    source=source.replace('int recipe,int pdl,void* stream)', 'int recipe,int pdl,int recipe_flags,void* stream)')
    source=source.replace('run<R,T,W,P>(a,pdl,(cudaStream_t)stream)','run<R,T,W,P>(a,pdl,recipe_flags,(cudaStream_t)stream)')
    # Clock64 boundaries are per-CTA deltas, not globally synchronized timestamps.
    source=source.replace('const int row0', 'const int row0')
    source=source.replace('B* sx=mem;','B* sx=mem;')
    module=load_inline(name='glue_audit_'+hashlib.sha256(source.encode()).hexdigest()[:16],
        cpp_sources='',cuda_sources=source,functions=None,is_python_module=False,no_implicit_headers=True,
        extra_cuda_cflags=['-O3','-std=c++17','-lineinfo'],verbose=False)
    lib=ctypes.CDLL(module);lib.glue_launch.argtypes=[ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_size_t]*2+[ctypes.c_int]*5+[ctypes.c_void_p]
    lib.glue_launch.restype=ctypes.c_int
    base=local_eval.load_module('glue_audit_best7',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
    weights,state=synthetic(base)
    fg=local_eval.load_module('glue_audit_dropin',HERE.parent/'dropin.py')
    flush=torch.empty(256<<20,device='cuda',dtype=torch.uint8);rows=[]
    for M in (1,8,16,32):
      stream=torch.randn((M,8192),device='cuda');local=torch.tensor([96],device='cuda',dtype=torch.int32)
      for mode in (1,2,0,3):
        op=fg.FusedGlue(state,M,mode,0,1024)
        x=torch.randn((M,op.k),device='cuda',dtype=torch.bfloat16)
        p=x.unsqueeze(0);s=1
        if mode==1:p=torch.randn((1,M,8192),device='cuda');s=1
        front=mode in (1,3)
        tensors=[state[op.name],x,p,stream,state['alpha1' if front else 'alpha2'],state['beta1' if front else 'beta2'],
          state['pre_ffn_w' if front else 'next_pre_attn_w'],state['post_ffn_w'],state['router_bias'],state['pre_expert_w'],
          local,op.parts,op.product,op.sout,op.hidden,op.x,op.ids,op.wts,op.lids,op.counts]
        ptrs=(ctypes.c_void_p*20)(*(t.data_ptr() for t in tensors))
        def call(flags,pdl=False):
          error=lib.glue_launch(ptrs,M,op.n,op.k,1024,s,p.stride(0),p.stride(1),int(p.dtype==torch.bfloat16),
            mode,0,int(pdl),flags,torch.cuda.current_stream().cuda_stream)
          if error:raise RuntimeError(f'CUDA{error}')
        timings={}
        for label,flags,pdl in [('full',0,False),('no_epilogue',2,False),('full_pdl',0,True)]+(
          [('no_prologue',1,False),('gemm_only',3,False)] if mode==1 else []):
          # Deleted norm work is replaced by finite zero shared activations;
          # the volatile MMA/control still execute. This is ineligible work deletion.
          timings[label]=measure(lambda:call(flags,pdl),flush)
          assert bool((op.counts==0).all())
        name={1:'pool_prologue',2:'pool_route',0:'unpool_boundary2',3:'o_boundary1'}[mode]
        r=dict(tokens=M,mode=name,grid=[op.n//64,op.k//1024,(M+7)//8],threads=128,
          final_ctas=(M+7)//8,token_warps=min(M,4),timings_us=timings)
        rows.append(r);print(json.dumps(r),flush=True)
    result=dict(kind='local work-deletion diagnostics; not candidates',source_sha256=hashlib.sha256(original.encode()).hexdigest(),
      gpu=torch.cuda.get_device_name(),rows=rows)
    (HERE/'mechanism_local.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
