"""Exercise both OL launch ABIs on sm120 with a native marker kernel and64KiB smem.

Only the Hopper arithmetic body and excessive local smem request are substituted;
configuration guards, tensor maps, dimensions and PDL launch setup remain real.
"""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'problems/muse_decode_layer'))
sp=importlib.util.spec_from_file_location('ew_wide_glue',H/'ALL_trial.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
ns=m._nf_vendor();cpp='#include <pybind11/pybind11.h>\n#include <cstdint>\nint ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\nint ew_attr();int ew_clear();'
x=torch.zeros((128,1024),device='cuda',dtype=torch.bfloat16);w=torch.zeros((8192,1024),device='cuda',dtype=torch.bfloat16);y=torch.zeros((128,8192),device='cuda',dtype=torch.bfloat16)
rows=[]
for label,marker in (('base',1),('unrenamed',2),('changed',3)):
    src=ns['OLAYER_CUDA']
    src=m._ew_old_wide(src) if label=='base' else m._nf_pdl_wide(src)
    if label=='unrenamed':src=src.replace('ol_projection_ew','ol_projection').replace('ol_launch_t_ew','ol_launch_t')
    ns['OLAYER_CUDA']=src;src=ns['olayer_source']([(128,128,4,3,1,1)])
    # Restore the vendor template so the next arm receives one port application.
    ns['OLAYER_CUDA']=m._nf_vendor()['OLAYER_CUDA']
    a=src.index('{',src.index('__global__'));b=src.index('\n}\n\nusing OLEncode',a)
    src=src[:a+1]+f'\n if(threadIdx.x==0 && blockIdx.x==0 && blockIdx.y==0)out[0]=__float2bfloat16({marker}.f);\n'+src[b:]
    src=src.replace('SMEM=MAIN>EPI?MAIN:EPI','SMEM=65536')
    kernel='ol_projection_ew' if label=='changed' else 'ol_projection'
    src+='\nint ew_clear(){return int(cudaGetLastError());}\nint ew_attr(){cudaFuncAttributes a{};int e=cudaFuncGetAttributes(&a,'+kernel+'<128,128,4,3,1,1,1>);return e?-e:a.maxDynamicSharedSizeBytes;}\n'
    mod=load_inline(name='ew_wide_glue_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=cpp,cuda_sources=src,functions=['ol_launch','ew_attr','ew_clear'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_120,code=sm_120'])
    y.zero_();rc=mod.ol_launch(0,x.data_ptr(),w.data_ptr(),y.data_ptr(),128,torch.cuda.current_stream().cuda_stream,0);mod.ew_clear();torch.cuda.synchronize()
    if label=='unrenamed':
        rows.append(dict(arm=label,returncode=rc,max_dynamic_smem=mod.ew_attr(),marker=y[0,0].item()));print(rows[-1],flush=True)
        continue
    assert rc==0 and y[0,0].item()==marker,(label,rc,y[0,0].item())
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rc=mod.ol_launch(0,x.data_ptr(),w.data_ptr(),y.data_ptr(),128,torch.cuda.current_stream().cuda_stream,0);assert rc==0
    g.replay();torch.cuda.synchronize();assert y[0,0].item()==marker
    rows.append(dict(arm=label,source_sha256=hashlib.sha256(src.encode()).hexdigest(),eager=True,capture=True,marker=marker,max_dynamic_smem=mod.ew_attr()));print(rows[-1],flush=True)
(H/'wide_glue_validation.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS original+changed O128 launch glue, eager+graph')
