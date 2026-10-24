"""Freeze the measured producer with optional clocks, plus a paired layer FA3 lab."""
import ast
import hashlib
from widen import widened
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def functions(path,names):
    src=path.read_text();tree=ast.parse(src);lines=src.splitlines(True);out=[]
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names:
            start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
            out.append(''.join(lines[start:n.end_lineno]))
    assert len(out)==len(names),(path,names,len(out))
    return '\n\n'.join(out)+'\n'

def common():
    ref=ROOT/'problems/muse_attention_decode/reference.py'
    src=ref.read_text();lines=src.splitlines(True);parts=[]
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name not in ('_fa3_scheduler_metadata','speed_of_light_ns'):
            parts.append(''.join(lines[n.lineno-1:n.end_lineno]))
    checks=functions(ROOT/'problems/muse_attention_decode/checks.py',{'compare','relative_l2','query_precision_bound'})
    return 'import math, time\nimport torch\nfrom torch.utils.cpp_extension import load_inline\nROW_FACTOR=5.0\nROW_NORM_FLOOR=.1\n'+checks+'\n'.join(parts)+'\n\ndef _fa3_scheduler_metadata(*args,**kwargs): return None\n'+(HERE/'common.py.inc').read_text()

def instrument():
    s=(ROOT/'problems/muse_attention_decode/dev/flashdecode/verify8.cu').read_text()
    s=s.replace('#include <stdexcept>','#include <stdexcept>\n#include <vector>')
    s=s.replace('int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;','int B,HQ,HK,stride,pages,ps,lg,ns,proj_splits; float scale;\n unsigned long long* clocks;')
    pos=s.index('template<int N,bool MASK,int VN>')
    s=s[:pos]+'''struct Trace { unsigned long long c[7]; };
DI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
'''+s[pos:]
    s=s.replace('template<int N,bool MASK,int VN>','template<int N,bool MASK,int VN,bool CLOCK>')
    s=s.replace('int vkey){','int vkey,Trace& trace){\n unsigned long long ts=0;if constexpr(CLOCK)ts=tick();')
    s=s.replace(' // Four lanes jointly own a query row.', ' if constexpr(CLOCK){const auto now=tick();trace.c[3]+=now-ts;ts=now;}\n // Four lanes jointly own a query row.')
    s=s.replace(' uint32_t packed[NT];',' if constexpr(CLOCK){const auto now=tick();trace.c[4]+=now-ts;ts=now;}\n uint32_t packed[NT];')
    s=s.replace(' for(int k=0;k<N/32;k++){',' for(int k=0;k<N/32;k++){')
    s=s.replace('#pragma unroll\n for(int k=0;k<N/32;k++){',' if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}\n#pragma unroll\n for(int k=0;k<N/32;k++){')
    s=s.replace('template<int ST,int KT>\n__global__','template<int ST,int KT,bool CLOCK>\n__global__',1)
    marker='}\ntemplate<int ST,int KT,bool CLOCK>\n__global__'
    assert marker in s;s=s.replace(marker,' if constexpr(CLOCK)trace.c[6]+=tick()-ts;\n'+marker,1)
    s=s.replace(' constexpr int KB=KT*64;',' Trace trace{};unsigned long long start=0,ts=0,init=0;if constexpr(CLOCK)start=tick();\n constexpr int KB=KT*64;',1)
    s=s.replace(' for(int it=0;it<tiles;it++){',' if constexpr(CLOCK)init=tick()-start;\n for(int it=0;it<tiles;it++){\n  if constexpr(CLOCK)ts=tick();',1)
    s=s.replace('  wait<ST-2>();__syncthreads();','  wait<ST-2>();__syncthreads();\n  if constexpr(CLOCK){const auto now=tick();trace.c[1]+=now-ts;ts=now;}',1)
    s=s.replace('  transpose_v<KT>', '  if constexpr(CLOCK){const auto now=tick();trace.c[0]+=now-ts;ts=now;}\n  transpose_v<KT>',1)
    s=s.replace('transpose_v<KT>(V+(it%ST)*KB,VT);__syncthreads();','transpose_v<KT>(V+(it%ST)*KB,VT);__syncthreads();\n  if constexpr(CLOCK)trace.c[2]+=tick()-ts;',1)
    for n,mask,vn in [(64,'false','KT'),(64,'true','KT'),(32,'true','32')]:
        s=s.replace(f'tile<{n},{mask},{vn}>(st,',f'tile<{n},{mask},{vn},CLOCK>(st,')
    s=s.replace('scale,sub*64);','scale,sub*64,trace);').replace('seq,scale,0);','seq,scale,0,trace);')
    marker='\n}\n// Each warp owns one query/head row.'
    stamp='''
 if constexpr(CLOCK){
  const unsigned long long elapsed=tick()-start;
  if(lane==0){
   auto* dst=p.clocks+((((int64_t)b*p.HK+h)*p.ns+split)*4+warp)*10;
#pragma unroll
   for(int k=0;k<7;k++)dst[k]=trace.c[k];
   dst[7]=init;dst[8]=elapsed;dst[9]=tiles;
  }
 }
'''
    assert marker in s;s=s.replace(marker,stamp+marker,1)
    s=s.replace('template<int ST,int KT>void launch','template<int ST,int KT,bool CLOCK>void launch')
    s=s.replace('verify8_main<ST,KT>','verify8_main<ST,KT,CLOCK>')
    s=s.replace('int64_t kt,int64_t stream){','int64_t kt,int64_t profile,int64_t clocks,int64_t stream){')
    s=s.replace('(int)ns,(int)proj_splits,(float)scale};','(int)ns,(int)proj_splits,(float)scale,(unsigned long long*)clocks};')
    old=''' if(kt==64){if(stages==3)launch<3,64>(p,s);else launch<4,64>(p,s);}
 else {if(stages==3)launch<3,128>(p,s);else launch<4,128>(p,s);}'''
    def branches(flag):return old.replace('launch<3,64>','launch<3,64,'+flag+'>').replace('launch<4,64>','launch<4,64,'+flag+'>').replace('launch<3,128>','launch<3,128,'+flag+'>').replace('launch<4,128>','launch<4,128,'+flag+'>')
    assert old in s;s=s.replace(old,' if(profile){'+branches('true')+'}else{'+branches('false')+'}',1)
    s+='''
namespace {
template<int ST,int KT,bool C>std::vector<int64_t> resources(){
 cudaFuncAttributes a;cudaFuncGetAttributes(&a,verify8_main<ST,KT,C>);
 constexpr int smem=ST*KT*128+8192+KT*64;
 cudaFuncSetAttribute(verify8_main<ST,KT,C>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
 int active=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,verify8_main<ST,KT,C>,128,smem);
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,smem,active};
}
}
std::vector<int64_t> verify8_resources(int64_t stages,int64_t kt,int64_t profile){
#define RES(S,K) if(stages==S && kt==K){return profile?resources<S,K,true>():resources<S,K,false>();}
 RES(3,64) RES(4,64) RES(3,128) RES(4,128)
#undef RES
 throw std::runtime_error("bad resource setting");
}
'''
    (HERE/'instrumented.cu').write_text(s)
    return s

def build():
    cuda=instrument()
    cpp='''#include <cstdint>
#include <vector>
void verify8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t,int64_t,int64_t);
std::vector<int64_t> verify8_resources(int64_t,int64_t,int64_t);
'''
    layer=ROOT/'problems/muse_decode_layer/cand_r3_best7_scale_recoded_m256.py'
    names={'_pdl_start','_pdl_wait','_launch','_qk_prep_kernel','qk_prep','_norm_gate_kernel','_attention_fa3_public','_best7_old_qk_prep_kernel','_best7_old_qk_prep'}
    layer_funcs=functions(layer,names)
    layer_header='\nimport triton\nimport triton.language as tl\nfrom sgl_kernel.flash_attn import flash_attn_with_kvcache\nHQ,HKV,HD=16,2,64\nQ_SIZE=1024\nPDL=True\n'
    (ROOT/'prof/lab/verify8_producer_phases.py').write_text(common()+layer_header+layer_funcs+'\nPHASE_CUDA='+repr(cuda)+'\nPHASE_CPP='+repr(cpp)+'\nWIDE_CUDA='+repr(widened(cuda))+'\n'+(HERE/'phases.py.inc').read_text())
    adapter=(ROOT/'problems/muse_decode_layer/dev/verify1_sliding_pick.py').read_text()
    # The adapter's code is embedded verbatim; its imports are all image packages.
    pair=common()+'''\nimport triton\nimport triton.language as tl\nfrom sgl_kernel.flash_attn import flash_attn_with_kvcache\nHQ,HKV,HD=16,2,64\nQ_SIZE=1024\nPDL=True\n'''+adapter+'\n'+layer_funcs+(HERE/'pair.py.inc').read_text()
    (ROOT/'prof/lab/verify1_sliding_fa3_pair.py').write_text(pair)
    print('payloads frozen',hashlib.sha256(cuda.encode()).hexdigest())

if __name__=='__main__':build()
