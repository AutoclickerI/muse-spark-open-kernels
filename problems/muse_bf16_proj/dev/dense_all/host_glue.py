"""Run exact Hopper C++ dispatch guards/arguments with empty local kernels."""
import importlib.util,hashlib,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('da_glue',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ns=m.da_vendor();captured=[];load=m.load_inline
def collect(name,cpp_sources='',cuda_sources='',**kw):
    captured.append((name,cpp_sources,cuda_sources,kw));return None
m.load_inline=collect;models=[]
for early in (False,True):
    for gate in (False,True):
        _,index,_=m.da_build_small(ns,early,gate);models.append((early,gate,index,captured[-1]))
ptr=torch.empty(16,device='cuda',dtype=torch.int32);x=torch.empty((32,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((32,8192),device='cuda');count=0
for early,gate,index,(name,cpp,source,kw) in models:
    a=source.index('{',source.index('v18s_gemv('));b=source.index('\nstatic int v18s_smem_optin()',a);source=source[:a]+'{ }\n'+source[b:]
    source=source.replace('  cudaLaunchKernelEx(&lc, kern,','  lc.attrs=nullptr;lc.numAttrs=0;\n  cudaLaunchKernelEx(&lc, kern,')
    tag=hashlib.sha256(source.encode()).hexdigest()[:12]
    mod=load(name='da_glue_'+tag,cpp_sources=cpp,cuda_sources=source,**kw)
    for (t,k,arm),(idx,cfg) in index.items():
        args=(idx,ptr.data_ptr(),x.data_ptr(),out.data_ptr(),ptr.data_ptr(),t,1024,k,*cfg,0,0,torch.cuda.current_stream().cuda_stream,0)
        if gate:args+=(x.data_ptr(),)
        rc=mod.v18s_launch(*args);assert rc==0,(early,gate,t,k,arm,rc);count+=1
insts=list(dict.fromkeys((bn or t,64,1,pf,2,True,2) for t in (128,256) for bn,s,pf,e in m.DA_WIDE));wide=0
for early in (False,True):
    source=m.da_wide_source(ns,insts,early);a=source.index('{',source.index('__global__'));b=source.index('\nusing EncodeFn=',a);source=source[:a]+'{ }\n'+source[b:]
    tag=hashlib.sha256(source.encode()).hexdigest()[:12]
    mod=load(name='da_wide_glue_'+tag,cpp_sources=ns['dd_decodediet_tt_mw_CPP'],cuda_sources=source,functions=['launch','info'],extra_cuda_cflags=['-O3',ns['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False)
    for t in (128,256):
        xx=torch.empty((t,8192),device='cuda',dtype=torch.bfloat16)
        for arm,(bn,s,pf,e) in enumerate(m.DA_WIDE):
            if e!=early:continue
            idx=insts.index((bn or t,64,1,pf,2,True,2))
            rc=mod.launch(idx,xx.data_ptr(),ptr.data_ptr(),ptr.data_ptr(),out.data_ptr(),t,2304,8192,s,0,torch.cuda.current_stream().cuda_stream);assert rc==0,(t,arm,rc);wide+=1
torch.cuda.synchronize();result=dict(passed=True,small_dispatches=count,wide_dispatches=wide,standin='only device bodies replaced; original shape guards, map encoding, sizes, pointer ABI and host launch executed; small cluster attribute removed for sm120')
(H/'host_glue_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
