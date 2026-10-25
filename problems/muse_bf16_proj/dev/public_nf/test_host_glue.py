"""Execute the real qkv16/32 C++ launch wrappers with empty CUDA stand-ins.
The launch interceptor validates every actual attribute; no Hopper arithmetic runs.
This covers the C++ stack/attribute path that a Python arithmetic stand-in bypasses.
"""
import hashlib,json,re
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
from port_body import _nf_vendor
H=Path(__file__).resolve().parent

def empty_kernels(src):
    # Match braces in C++ code while ignoring strings and comments.
    def closing(start):
        i=start;depth=0;quote=None;comment=None
        while i<len(src):
            c=src[i];n=src[i:i+2]
            if comment=='line':
                if c=='\n':comment=None
            elif comment=='block':
                if n=='*/':comment=None;i+=1
            elif quote:
                if c=='\\':i+=1
                elif c==quote:quote=None
            elif n=='//':comment='line';i+=1
            elif n=='/*':comment='block';i+=1
            elif c in ('"',"'"):quote=c
            elif c=='{':depth+=1
            elif c=='}':
                depth-=1
                if depth==0:return i+1
            i+=1
        raise AssertionError('C++ unmatched brace')
    spans=[];i=0
    while True:
        a=src.find('__global__',i)
        if a<0:break
        a=src.index('{',a);b=closing(a);spans.append((a,b));i=b
    for a,b in reversed(spans):src=src[:a]+'{}'+src[b:]
    return src

HEADER='''#include <cuda_runtime.h>
#include <stdexcept>
template<typename F,typename...A> static cudaError_t nf_verify_launch(const cudaLaunchConfig_t* c,F f,A...a) {
  if(c->numAttrs!=3)throw std::runtime_error("expected cluster, policy, PDL");
  unsigned bits=0;
  for(int i=0;i<c->numAttrs;++i) {
    auto id=c->attrs[i].id;
    if(id==cudaLaunchAttributeClusterDimension) {bits|=1;if(c->attrs[i].val.clusterDim.y!=4)throw std::runtime_error("cluster4");}
    else if(id==cudaLaunchAttributeClusterSchedulingPolicyPreference)bits|=2;
    else if(id==cudaLaunchAttributeProgrammaticStreamSerialization) {bits|=4;if(c->attrs[i].val.programmaticStreamSerializationAllowed!=1)throw std::runtime_error("PDL");}
    else throw std::runtime_error("unexpected launch attribute");
  }
  if(bits!=7)throw std::runtime_error("missing launch attribute");
  return cudaSuccess;
}
'''
ns=_nf_vendor()
def compile_standin(name,cpp_sources='',cuda_sources='',**kw):
    src=HEADER+empty_kernels(cuda_sources).replace('cudaLaunchKernelEx(','nf_verify_launch(')
    return load_inline(name='nf_host_glue_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources=cpp_sources,cuda_sources=src,functions=kw['functions'],
        extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],extra_cflags=['-O2','-fstack-protector-all'],no_implicit_headers=True,with_pytorch_error_handling=False)
ns['load_inline']=compile_standin
rows=[]
for family,build,symbol,m in [('q16','b2q_tt_v10_m32x__q_build','r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch',16),('q32','tt_v10_m32x__q_build','m32x_q_launch',32)]:
    mod=ns[build]();fn=getattr(mod,symbol)
    for n in (1280,2304):
        x=torch.zeros((m,8192),device='cuda',dtype=torch.bfloat16);out=torch.empty((m,n),device='cuda');dummy=torch.empty(1,device='cuda',dtype=torch.int32)
        rc=fn(0,dummy.data_ptr(),x.data_ptr(),out.data_ptr(),dummy.data_ptr(),m,n,8192,1,4,8,4,0,0,torch.cuda.current_stream().cuda_stream,0)
        assert rc==0,(family,n,rc)
        rows.append(dict(family=family,N=n,M=m,attributes=3,rc=rc));print('PASS',rows[-1],flush=True)
(H/'host_glue_validation.json').write_text(json.dumps(dict(passed=True,adapter_sha256=hashlib.sha256((H/'port_body.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
