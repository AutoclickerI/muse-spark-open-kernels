"""Explicit local-only launch stand-ins; never embedded or shipped to the judge.

SM120 cannot launch the selected Hopper attention binaries. Exercise their Python
dispatch, allocation, counter/partial ABI and graph capture with admitted arithmetic.
Exact changed CUDA still goes through the sm90a static compilation checks.
"""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

H=Path(__file__).resolve().parent
BASE=H.parents[1]/'cand_r3_best7_s14_final.py'


def install(module):
    m=module
    if m.torch.cuda.get_device_capability()==(9,0):return {}
    result={}
    original={target.id:ast.literal_eval(n.value) for n in ast.parse(BASE.read_text()).body
              if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
              for target in n.targets if isinstance(target,ast.Name)}
    if m.VQ8_CUDA!=original['VQ8_CUDA']:
        vq8_cached=[]
        def vq8_module():
            if not vq8_cached:
                arch=''.join(map(str,m.torch.cuda.get_device_capability()))
                vq8_cached.append(m.load_inline(name='muse_vq8_c262e4c5f9c3_'+arch,cpp_sources=m.VQ8_CPP,
                    cuda_sources=original['VQ8_CUDA'],functions=['vq8_forward','vq8_resources'],
                    extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False))
            return vq8_cached[0]
        m._vq8_module=vq8_module
        result['regmax']='frozen S14 N64 launch stands in for Hopper N128 ring; Python stage4 selection retained'
    if hasattr(m,'cs_attention'):
        cs_cached=[];context={};original_attention=m.cs_attention
        cuda='''#include <cuda_runtime.h>
#include <cstdint>
__global__ void cp(const uint16_t* x,uint16_t* y,int n,int* cnt,int groups){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=x[i];if(i<groups*32)cnt[i]=0;
}
void copy(int64_t x,int64_t y,int64_t n,int64_t cnt,int64_t groups,int64_t stream){
  cp<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((uint16_t*)x,(uint16_t*)y,n,(int*)cnt,groups);
}
'''
        def cs_module():
            if not cs_cached:
                arch=''.join(map(str,m.torch.cuda.get_device_capability()))
                tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
                mod=m.load_inline(name='s15_cs_standin_'+tag+'_'+arch,cpp_sources='#include <cstdint>\n#include <pybind11/pybind11.h>\nvoid copy(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',
                    cuda_sources=cuda,functions=['copy'],extra_cuda_cflags=['-O3',f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                    no_implicit_headers=True,with_pytorch_error_handling=False)
                def forward(*args):
                    assert len(args)==30
                    out=context['output'];groups=args[11]*args[14]*args[21]
                    mod.copy(out.data_ptr(),args[10],out.numel(),args[-2],groups,args[-4])
                cs_cached.append(SimpleNamespace(sliding_resources=lambda *a:[0,0,53504,1,0],sliding_forward=forward))
            return cs_cached[0]
        def cs_attention(data,qkv=None,**kwargs):
            context['output']=m.attention_partial_port(qkv, data, False)
            return original_attention(data,qkv,**kwargs)
        m.cs_module=cs_module;m.cs_attention=cs_attention
        # prepare compiles modules on worker threads. Finish this local-only module
        # before those workers can request it concurrently or hide a build error.
        cs_module()
        result['last']='actual cs_attention Python ABI; direct-partials arithmetic plus native pointer-copy/counter-reset launch'
    print('S15_LOCAL_STANDINS',result,flush=True)
    return result
