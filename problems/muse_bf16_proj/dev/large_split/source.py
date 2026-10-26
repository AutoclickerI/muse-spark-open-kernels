"""Matched raw WGMMA mechanism oracles; no synthetic oracle is a candidate."""
import ast,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
T=R/'problems/muse_bf16_proj/dev/dense_all/runtime.py.inc'
node=next(n for n in ast.parse(T.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='da_wide_source')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(T),'exec'))
CONFIG={(128,'qkv_gate'):(128,8),(128,'o'):(128,2),(128,'pool_router'):(128,4),(128,'unpooler'):(128,2),(256,'qkv_gate'):(128,4),(256,'o'):(256,2),(256,'pool_router'):(128,8),(256,'unpooler'):(256,2)}
MODES=('raw','smem_x','mathfree','no_epilogue','x_mathfree','x_no_ep','mathfree_no_ep','x_mathfree_no_ep','bf16')
INSTANCES=[(bn,64,1,2,2,False,2) for bn in (128,256)]
CPP='''#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
int launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);
std::vector<int64_t> resource(int,int);
'''
def source(ns,mode):
    assert mode in MODES
    s=da_wide_source(ns,INSTANCES,False)
    if mode in ('smem_x','x_mathfree','x_no_ep','x_mathfree_no_ep'):
        s=s.replace('waitbar(full+s*8,(i/Q)&1);','waitbar(full+s*8,0);')
        s=s.replace('      loadx(i+Q);','      // Oracle: reuse the first Q real activation tiles; not a GEMM candidate.')
    if 'mathfree' in mode:
        for bn in (16,32,64,128,256):
            # Retain every raw weight fragment as an observable volatile global
            # store in a separate sink, preserving loaded bytes without MMA.
            old=f'      if constexpr(BN=={bn})mma{bn}(acc,a,desc(base+s*SB));'
            if old in s:s=s.replace(old,'')
        needle='      int s=i%Q;\n      waitbar('
        assert s.count(needle)==1
        s=s.replace(needle,'''      if(tid<128) {
        uint32_t h=0;
        #pragma unroll
        for(int j=0;j<16;++j)h^=a[j];
        ((volatile uint32_t*)esc)[(blockIdx.y*gridDim.x+blockIdx.x)*128+tid]=h;
      }
'''+needle)
        s=s.replace('asm volatile("mov.b32 %0,0;":"=f"(acc[j]));','{uint32_t clock_bits;asm volatile("mov.u32 %0,%%clock;":"=r"(clock_bits));acc[j]=__uint_as_float(clock_bits);}')
    if mode in ('no_epilogue','x_no_ep','mathfree_no_ep','x_mathfree_no_ep'):
        start=s.index('  int row=warp*16+lane/4,col=2*(lane%4);');end=s.index('\n}\n',start)
        s=s[:start]+'''  float checksum=0;
  #pragma unroll
  for(int j=0;j<BN/2;++j)checksum+=acc[j];
  ((volatile float*)out)[(blockIdx.y*gridDim.x+blockIdx.x)*128+tid]=checksum;
'''+s[end:]
    if mode=='bf16':
        old='    *(float4*)(out+(long)(mt*BN+token)*n+nt*R+row)=sum;'
        new='''    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)((__nv_bfloat16*)out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);'''
        assert old in s;s=s.replace(old,new)
    if 'mathfree' in mode:
        # A real WGMMA rendezvous prevents the producer warp from lapping
        # other warps across an mbarrier parity cycle. Keep that ordering when
        # removing math; every thread must have observed the old completion.
        needle='      waitbar(full+s*8,'
        start=s.index(needle);end=s.index(';',start)+1
        s=s[:end]+'\n      __syncthreads();'+s[end:]
    import re
    wait=r'D void waitbar\(uint32_t b,int p\) \{.*?\n\}'
    guard='D void waitbar(uint32_t b,int p) {\n  unsigned long long started=clock64();int ready;\n  do {\n    asm volatile("{.reg .pred q; mbarrier.try_wait.parity.shared::cta.b64 q,[%1],%2; selp.u32 %0,1,0,q;}":"=r"(ready):"r"(b),"r"(p):"memory");\n    if(!ready && clock64()-started>2000000ull)asm volatile("trap;");\n  }while(!ready);\n}'
    s,count=re.subn(wait,lambda match:guard,s,count=1,flags=re.S);assert count==1
    # tab is unused for raw BF16. Use it only as a declared shared-memory
    # resource constraint to match math-free occupancy to its ordinary twin.
    s=s.replace('b=WK*BN*(R+4)*4,smem=a>b?a:b;', 'b=WK*BN*(R+4)*4,natural=a>b?a:b;\n  int smem=natural>int(tab)?natural:int(tab);')
    s=s.replace('static bool ready=false;\n  if(!ready) {','static int ready=0;\n  if(ready<smem) {').replace('    ready=true;','    ready=smem;')
    # Query real register/local/shared resources and active blocks for each body.
    pos=s.index('int info(int id)')
    s=s[:pos]+'''std::vector<int64_t> resource(int id,int pad){
  cudaFuncAttributes a{};int blocks=0,maxsm=0;cudaDeviceGetAttribute(&maxsm,cudaDevAttrMaxSharedMemoryPerMultiprocessor,0);
#define RES(B) {constexpr int natural=(2*(B*128+8)+1023)/1024*1024 > B*68*4?(2*(B*128+8)+1023)/1024*1024:B*68*4;int smem=pad>natural?pad:natural;auto f=p10_daw_normal_gemm<B,64,1,2,2,false,2>;cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);cudaFuncGetAttributes(&a,f);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,f,128,smem);return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,blocks,maxsm};}
  if(id==0){RES(128)}else{RES(256)}
#undef RES
}
'''
    return '#include <vector>\n'+s.replace('p10_daw_normal','p10_ls_'+mode)
