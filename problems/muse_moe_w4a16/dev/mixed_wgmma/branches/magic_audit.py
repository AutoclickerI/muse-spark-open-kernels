"""Exhaust all FP4 pairs x finite E4M3 scale codes on the local GPU.

The baseline is the shipped BF16 subnormal placement and HMUL2. The FP16
subnormal placement is tested both as raw bits and after a real BF16 conversion.
The INT4 magic-number recipe is also tested against E2M1 (different formats).
"""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
__device__ unsigned pair(unsigned a,unsigned b){return a|(b<<16);}
__global__ void audit(const unsigned* q,const unsigned* sf,const unsigned* hs,
                     unsigned* out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 unsigned v=q[i],s=sf[i],h=hs[i];
 unsigned b=((v&0x00070007u)<<6)|((v&0x00080008u)<<12);
 unsigned f=((v&0x00070007u)<<9)|((v&0x00080008u)<<12),want,got;
 asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(want):"r"(b),"r"(s));
 asm("mul.rn.f16x2 %0,%1,%2;":"=r"(got):"r"(f),"r"(h));
 __half2 hp=*reinterpret_cast<__half2*>(&got);
 __nv_bfloat162 cv=__floats2bfloat162_rn(__half2float(__low2half(hp)),__half2float(__high2half(hp)));
 unsigned mag=(v&0x00070007u)|0x64006400u,integer;
 // Standard INT4 mantissa magic: (1024 + nibble) - 1024, then /2.
 asm("sub.rn.f16x2 %0,%1,%2;":"=r"(integer):"r"(mag),"r"(0x64006400u));
 unsigned iscale;
 asm("mul.rn.f16x2 %0,%1,%2;":"=r"(iscale):"r"(integer),"r"(0x38003800u));
 out[4*i]=want;out[4*i+1]=got;out[4*i+2]=*reinterpret_cast<unsigned*>(&cv);out[4*i+3]=iscale;
}
extern "C" int run(const void*q,const void*s,const void*h,void*out,int n,void*st){
 audit<<<(n+255)/256,256,0,(cudaStream_t)st>>>((const unsigned*)q,(const unsigned*)s,(const unsigned*)h,(unsigned*)out,n);
 return (int)cudaGetLastError();
}
'''
CPP = '#include <pybind11/pybind11.h>\n#include <cstdint>\nextern "C" int run(const void*,const void*,const void*,void*,int,void*);'


def main():
    dev='cuda';codes=torch.arange(256,device=dev,dtype=torch.uint8)
    scales=codes.view(torch.float8_e4m3fn).float()
    scales=scales[torch.isfinite(scales)]
    a=scales.abs();nz=a>0
    mant,exp=torch.frexp(torch.where(nz,a,torch.ones_like(a)))
    encoded=torch.where(nz,((exp-1+21).int()<<3)|((mant*2-1)*8).round().int(),0)
    sf16=(encoded<<4)+0x7000
    hs=(a*64).half().view(torch.int16).int()&0xffff
    p=torch.arange(256,device=dev,dtype=torch.int32)
    lo=(p&15)[None,:].expand(len(scales),-1)
    hi=(p>>4)[None,:].expand_as(lo)
    neg=(scales<0).int()[:,None]*8
    lo=torch.where(nz[:,None],lo^neg,0);hi=torch.where(nz[:,None],hi^neg,0)
    q=(lo|(hi<<16)).flatten().contiguous()
    ss=(sf16|(sf16<<16))[:,None].expand(-1,256).flatten().contiguous()
    hh=(hs|(hs<<16))[:,None].expand(-1,256).flatten().contiguous()
    out=torch.empty((q.numel(),4),device=dev,dtype=torch.int32)
    mod=load_inline(name='p3b_magic_'+hashlib.sha256(CUDA.encode()).hexdigest()[:12],cpp_sources=CPP,cuda_sources=CUDA,
                    functions=['run'],no_implicit_headers=True,with_pytorch_error_handling=False,
                    extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_120,code=sm_120'])
    import ctypes
    fn=ctypes.CDLL(mod.__file__).run
    fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
    assert fn(q.data_ptr(),ss.data_ptr(),hh.data_ptr(),out.data_ptr(),q.numel(),torch.cuda.current_stream().cuda_stream)==0
    torch.cuda.synchronize()
    raw=(out[:,0]!=out[:,1]);converted=(out[:,0]!=out[:,2])
    # Unscaled INT4 recipe only represents E2M1 magnitudes 0 through 4 correctly.
    vals=torch.tensor([0,.5,1,1.5,2,3,4,6],device=dev,dtype=torch.float16)
    wantmag=vals[(p&7)].view(torch.int16).int()&0xffff
    one=int((scales==1).nonzero()[0]);integ=out[one*256:(one+1)*256,3]&0xffff
    result=dict(pairs=q.numel(),finite_scale_codes=len(scales),raw_fp16_bits_mismatch=int(raw.sum()),
                fp16_converted_to_bf16_mismatch=int(converted.sum()),
                int4_magic_magnitude_mismatch=int((integ!=wantmag).sum()),
                core_cuda_sha256=hashlib.sha256(CUDA.encode()).hexdigest())
    assert result['fp16_converted_to_bf16_mismatch']==0,result
    (Path(__file__).parent/'magic_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':main()
