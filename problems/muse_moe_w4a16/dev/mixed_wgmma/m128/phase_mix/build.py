"""Compose exact admitted producers using one padded route layout and phase dispatch."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;M128=HERE.parent;ROOT=HERE.parents[5]
sys.path[:0]=[str(M128.parent),str(ROOT/'tools')]
from build_prefill_lab import subset
from native_abi import CPP_PRELUDE
from okc_run import rejected_words


def nt_phase():
    s=(M128/'issue_limit/norig.cu').read_text()
    # All weight transport, unpack, BF16x2 scale and MMA instructions are the
    # recoded NT1 implementation. Move routing to the common route kernel and
    # expose each phase; the small specialization owns exactly count<=16.
    s=s.replace('namespace norig','namespace pmnt').replace('norig::','pmnt::')
    a=s.index('template <int NT>\n__global__');b=s.index('size_t smem_bytes(',a)
    original=s[a:b]
    head='''template<int NT,bool FIRST,bool SMALL>
__global__ void __launch_bounds__(NTHR,2) phase(
 const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,
 const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,
 __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){
 constexpr int NB=8*NT,RED=NT*4;
 extern __shared__ __align__(128) uint8_t smem[];
 uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);
 uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;
 Route*rt=(Route*)(emptyb+nstage);
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;
 if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
 if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>16)nb=0;
  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;
  int u1=nb*NT1,u2=nb*NT2;
  #pragma unroll
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}
  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;
  if(lane==0){rt->us1[0]=rt->us2[0]=0;}
  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);
  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}
 }
 __syncthreads();
'''
    p=original.index('  const int U1 = rt->us1[NLOC]');q=original.index('  // ---------------- consumers: deterministic')
    setup=original[p:q].replace('const int first_count=b1-a1, total_count=first_count+b2-a2;',
                               'const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;')
    p=original.index('  const int q = lane & 3;');q=original.index('  if (cur_pass == 0 && seg_tiles > 0) {\n',p)
    loop=original[p:q]
    loop=loop.replace('for (int pass = 0; pass < 2; ++pass)', 'for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass)')
    p=loop.index('        if (cur_pass == 0 && seg_tiles > 0)');q=loop.index('#pragma unroll\n        for (int j = 0;',p)
    loop=loop[:p]+'''        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;
'''+loop[q:]
    loop=loop.replace('pair_of[pos] >> 3','SLOTS[e*M+idx] >> 3')
    loop=loop.replace('const float h = G1 / (1.f + __expf(-G1)) * U;',
        'G1=__bfloat162float(__float2bfloat16_rn(G1));\n            U=__bfloat162float(__float2bfloat16_rn(U));\n            const float h = G1 / (1.f + __expf(-G1)) * U;')
    loop=loop.replace('U *= sc;','U *= gs13[e+32]*256.f;')
    loop=loop.replace('if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;',
        'if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];')
    loop=loop.replace('atomicAdd(ss + off + idx, t2);','atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);')
    # Make the blend cut already admitted by the small-M lane.
    s=s[:a]+head+setup+loop+'}\n\n'+s[b:]
    s=s.replace('const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);',
        'uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));')
    s=s[:s.index('int64_t norig_init()')]
    s+='''
int64_t pm_nt_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 auto set=[&]<bool F,bool S>(){cudaFuncAttributes a{};WX_CUDA_CHECK(cudaFuncGetAttributes(&a,pmnt::phase<1,F,S>));WX_CUDA_CHECK(cudaFuncSetAttribute(pmnt::phase<1,F,S>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};
 set.template operator()<true,false>();set.template operator()<false,false>();set.template operator()<true,true>();set.template operator()<false,true>();return opt;}
int64_t pm_nt_smem(int64_t stages,int64_t M){return pmnt::smem_bytes(1,stages,M);}
int64_t pm_nt_resident(int64_t stages,int64_t M,int64_t first,int64_t small){int n;
 auto get=[&]<bool F,bool S>(){WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmnt::phase<1,F,S>,pmnt::NTHR,pmnt::smem_bytes(1,stages,M)));};
 if(first){if(small)get.template operator()<true,true>();else get.template operator()<true,false>();}else{if(small)get.template operator()<false,true>();else get.template operator()<false,false>();}return n;}
void pm_nt_run(std::vector<int64_t>v,int64_t first,int64_t small,int64_t M,int64_t stages,int64_t grid,int64_t stream){
 auto launch=[&]<bool F,bool S>(){pmnt::phase<1,F,S><<<grid,pmnt::NTHR,pmnt::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(
 (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};
 if(first){if(small)launch.template operator()<true,true>();else launch.template operator()<true,false>();}else{if(small)launch.template operator()<false,true>();else launch.template operator()<false,false>();}WX_CUDA_CHECK(cudaGetLastError());}
'''
    assert 'pair_of' not in s and 'C_DONE +' not in s
    return s


def meta_source():
    return '''
#include <cuda_runtime.h>
#include <vector>
__global__ void pm_meta_kernel(const int*C,const int*ORDER,int*T32,int*T128,int*T16,float*SS,int cap){
 int t=threadIdx.x,idx=blockIdx.x*blockDim.x+t;
 for(int i=idx;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v32=n>16?(n+31)/32:0,v128=(n+127)/128,v16=(n+15)/16;
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v32,d),b=__shfl_up_sync(~0u,v128,d),c=__shfl_up_sync(~0u,v16,d);if(t>=d){v32+=a;v128+=b;v16+=c;}}
  T32[t+1]=v32;T128[t+1]=v128;T16[t+1]=v16;if(t==0){T32[0]=0;T128[0]=0;T16[0]=0;}
 }
}
void pm_meta(std::vector<int64_t>v,int64_t cap,int64_t stream){pm_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(float*)v[5],cap);}
'''


def main():
    spec=importlib.util.spec_from_file_location('onebuilder',M128/'one_read/build.py');one=importlib.util.module_from_spec(spec);spec.loader.exec_module(one)
    codes={p:(M128/'issue_limit'/f'{p}.cu').read_text() for p in ('idist','norig')}
    codes['or_rs']=one.rs_source();codes['pm_nt']=nt_phase();codes['pm_meta']=meta_source()
    cpp=CPP_PRELUDE;funs=[]
    cpp+='int64_t idist_init();std::vector<int64_t> idist_info(int64_t,int64_t,int64_t);void idist_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void idist_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);\n'
    funs += ['idist_'+f for f in ('init','info','route','run','finish')]
    cpp+='int64_t or_rs_init();std::vector<int64_t> or_rs_info(int64_t,int64_t);void or_rs_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
    funs += ['or_rs_'+f for f in ('init','info','run')]
    cpp+='int64_t norig_init();int64_t norig_smem(int64_t,int64_t);int64_t norig_resident(int64_t,int64_t);void norig_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);\n'
    funs += ['norig_'+f for f in ('init','smem','resident','run')]
    cpp+='int64_t pm_nt_init();int64_t pm_nt_smem(int64_t,int64_t);int64_t pm_nt_resident(int64_t,int64_t,int64_t,int64_t);void pm_nt_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void pm_meta(std::vector<int64_t>,int64_t,int64_t);\n'
    funs += ['pm_nt_init','pm_nt_smem','pm_nt_resident','pm_nt_run','pm_meta']
    s=subset((ROOT/'prof/lab/moe_m128_issue_limit.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph','issue_prepare','issue_word_bank','issue_nt_prepare'))
    tree=ast.parse(s)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name in ('issue_module','_v9_module'))
        and not (isinstance(n,ast.Assign) and any(getattr(t,'id','') in ('ISSUE_CUDA','ISSUE_CPP','ISSUE_FUNCTIONS','ISSUE_OPTS') for t in n.targets))]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='issue_module':n.func.id='pm_module'
    # Exact baseline functions preserve the D32g call/allocations and NT1rec.
    extra=subset((ROOT/'prof/lab/moe_m128_issue_limit.py').read_text(),('issue_plan','issue_call','issue_nt_call'))
    et=ast.parse(extra);keep={'issue_plan','issue_call','issue_nt_call'}
    for n in et.body:
        if isinstance(n,ast.FunctionDef) and n.name in keep:
            for v in ast.walk(n):
                if isinstance(v,ast.Name) and v.id=='issue_module':v.id='pm_module'
    baseline=ast.unparse(ast.Module(body=[n for n in et.body if isinstance(n,ast.FunctionDef) and n.name in keep],type_ignores=[]))
    cuda='\n'.join(codes.values())
    for p,c in codes.items():(HERE/(p+'.cu')).write_text(c)
    text=ast.unparse(tree)+'\nimport functools\nimport gc\nfrom torch.utils.cpp_extension import load_inline\n'
    text+='PM_CUDA='+repr(cuda)+'\nPM_CPP='+repr(cpp)+'\nPM_FUNCTIONS='+repr(funs)+'\n'+baseline+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    out=ROOT/'prof/lab/moe_m128_phase_mix.py';out.write_text(text)
    manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),sources={p:hashlib.sha256(c.encode()).hexdigest() for p,c in codes.items()},
        control='exact dc3fea0a D32g, config1 grid1056',one_read_receipt='640d67e7e56d42e5',one_read='best RS64g2112 (361.16us; rejected vs290.83)',
        phase_layout='one C/OFF/SLOTS/POS with padded32 expert rows; separate task prefixes for NT32 count>16 and NT128; no host route decisions',
        banks='uint8 recoded NT1 and one-read; u16 exact descriptor D32; no prepare expanded weights')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(out.relative_to(ROOT),manifest['sha256'],len(text))


if __name__=='__main__':main()
