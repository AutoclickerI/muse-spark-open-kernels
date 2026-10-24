"""Generate a native Q-major M64 WGMMA producer with the existing parallel combine."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def cuda_source():
    base=(ROOT/'problems/muse_attention_decode/dev/producer_phases/instrumented.cu').read_text()
    prefix=base[:base.index('// One CTA transposes V once per stage')]
    prefix=prefix.replace('((r&3))','(((r>>1)&3))').replace('((r&3))*16','(((r>>1)&3))*16').replace('^(r&3)','^((r>>1)&3)')
    assert '((r>>1)&3)' in prefix
    trace=base[base.index('struct Trace'):base.index('template<int N,bool MASK,int VN,bool CLOCK>')]
    trace=trace.replace('unsigned long long c[7]','uint32_t c[7]')
    body=(HERE/'body.cu.inc').read_text()
    regs=','.join('%'+str(i) for i in range(32));outputs=','.join(f'"+f"(d[{i//4}][{i%4}])' for i in range(32))
    wrapper=f'''DI void wg8(float (&d)[8][4],uint64_t a,uint64_t b){{
#if USE_WGMMA
 asm volatile("{{.reg .pred p;setp.ne.b32 p,1,0;wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {{{regs}}},%32,%33,p,1,1;}}":{outputs}:"l"(a),"l"(b));
#endif
}}
'''
    body=body.replace('WG_WRAPPERS',wrapper)
    combine=base[base.index('// Each warp owns one query/head row.'):base.index('template<int ST,int KT,bool CLOCK>void launch')]
    host='''
template<int ST,bool C>void native_launch(VP p,cudaStream_t stream){
 constexpr int smem=ST*8192+20480;
 cudaFuncSetAttribute(native_verify8<ST,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 native_verify8<ST,C><<<dim3(p.ns,p.HK,p.B),256,smem,stream>>>(p);
 verify8_combine<<<dim3(16,p.HK,p.B),128,0,stream>>>(p);
}
}
void native_forward(int64_t qkv,int64_t loc,int64_t pt,int64_t seq,int64_t kc,int64_t vc,
 int64_t ks,int64_t vs,int64_t parts,int64_t out,int64_t B,int64_t HQ,int64_t HK,int64_t stride,
 int64_t pages,int64_t ps,int64_t ns,int64_t proj_splits,double scale,int64_t stages,int64_t profile,int64_t clocks,int64_t stream){
 VP p{(const __nv_bfloat16*)qkv,(const int64_t*)loc,(const int*)pt,(const int*)seq,(uint8_t*)kc,(uint8_t*)vc,
 (const float*)ks,(const float*)vs,(float*)parts,(__nv_bfloat16*)out,(int)B,(int)HQ,(int)HK,(int)stride,(int)pages,
 (int)ps,0,(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};
 while((1<<p.lg)<ps)p.lg++;
 auto stream_ptr=(cudaStream_t)stream;
 if(stages==4){if(profile)native_launch<4,true>(p,stream_ptr);else native_launch<4,false>(p,stream_ptr);}
 else if(stages==8){if(profile)native_launch<8,true>(p,stream_ptr);else native_launch<8,false>(p,stream_ptr);}
 else throw std::runtime_error("native stages must be 4 or 8");
 auto error=cudaGetLastError();if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
namespace {
template<int ST,bool C>std::vector<int64_t> native_resource(){
 cudaFuncAttributes a;cudaFuncGetAttributes(&a,native_verify8<ST,C>);
 constexpr int smem=ST*8192+20480;
 cudaFuncSetAttribute(native_verify8<ST,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,native_verify8<ST,C>,256,smem);
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};
}
}
std::vector<int64_t> native_resources(int64_t stages,int64_t clocks){
 if(stages==4)return clocks?native_resource<4,true>():native_resource<4,false>();
 return clocks?native_resource<8,true>():native_resource<8,false>();
}
'''
    source=prefix+trace+body+combine+host
    (HERE/'native.cu').write_text(source)
    return source

def build():
    source=cuda_source()
    cpp='''#include <cstdint>
#include <vector>
void native_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t);
std::vector<int64_t> native_resources(int64_t,int64_t);
'''
    driver=(HERE/'driver.py.inc').read_text()
    adapter='''"""Native Hopper full-q8 attention experiment. H200 promotion is pending."""
import torch
import hashlib
from torch.utils.cpp_extension import load_inline
'''+ '\nNATIVE_CUDA='+repr(source)+'\nNATIVE_CPP='+repr(cpp)+'\n'+driver
    (HERE/'runtime.py.inc').write_text('\nNATIVE_CUDA='+repr(source)+'\nNATIVE_CPP='+repr(cpp)+'\nimport hashlib\n'+driver)
    (ROOT/'problems/muse_decode_layer/dev/verify8_native_wgmma.py').write_text(adapter)
    print('native adapter generated')
    return source,cpp,driver

if __name__=='__main__':build()
