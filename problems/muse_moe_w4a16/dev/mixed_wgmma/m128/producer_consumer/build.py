"""Expert-wide pipelined shared-A WGMMA, with admitted H_H/NT1 controls."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;BASE=HERE.parent/'phase_mix';ROOT=HERE.parents[5]
sys.path[:0]=[str(HERE.parents[1]),str(ROOT/'tools')]
from build_prefill_lab import subset
from native_abi import CPP_PRELUDE
from okc_run import rejected_words


def source():
    s=(HERE.parent/'one_read/or_ss128.cu').read_text()
    a=s.index('template<int NT,bool OVERLAP');b=s.index('template<int NW>');c=s.index('template<int NT,int WIDTH',b)
    s=s[:a]+s[b:c]+'}\n'
    s=s.replace('or_ss128','pc').replace('int t=threadIdx.x,w=t>>5,l=t&31;unsigned a[4];','int t=threadIdx.x&127,w=t>>5,l=t&31;unsigned a[4];')
    old='unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;'
    assert old in s;s=s.replace(old,'unsigned w,mask=0x81c081c0u;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));')
    return s+(HERE/'kernel.cu.inc').read_text()


def main():
    cuda=source();(HERE/'pc.cu').write_text(cuda)
    cpp=CPP_PRELUDE+'int64_t pc_init();std::vector<int64_t> pc_info(int64_t,int64_t,int64_t);void pc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void pc_meta(std::vector<int64_t>,int64_t,int64_t,int64_t);\n'
    cuda+='''
__global__ void pc_meta_kernel(const int*C,const int*ORDER,int*T,int*T32,float*SS,int cap,int cut){
 int t=threadIdx.x,all=blockIdx.x*blockDim.x+t;
 for(int i=all;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v=n>cut,b=v?(n+31)/32:0;
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),c=__shfl_up_sync(~0u,b,d);if(t>=d){v+=a;b+=c;}}
  T[t+1]=v;T32[t+1]=b;if(t==0){T[0]=0;T32[0]=0;}}
}
void pc_meta(std::vector<int64_t>v,int64_t cap,int64_t cut,int64_t stream){pc_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(float*)v[4],cap,cut);WX_CUDA_CHECK(cudaGetLastError());}
'''
    (HERE/'pc.cu').write_text(cuda)
    helpers=subset((ROOT/'prof/lab/moe_m128_phase_mix.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph','pm_error'))
    hh=(BASE/'hh_cell.py').read_text();assert hashlib.sha256(hh.encode()).hexdigest()=='970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f'
    small=(ROOT/'problems/muse_decode_layer/dev/nt1_branches/nt1_lop3_cell.py').read_text()
    text=helpers+'\n'+hh+'\n'+small+'\nimport gc\nimport time\nimport statistics\n'
    text+='PC_CUDA='+repr(cuda)+'\nPC_CPP='+repr(cpp)+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    out=ROOT/'prof/lab/moe_m128_producer_consumer.py';out.write_text(text)
    manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        control_HH='770809cf159f41e2',H_H_sha256=hashlib.sha256(hh.encode()).hexdigest(),small_cell_sha256=hashlib.sha256(small.encode()).hexdigest(),
        prerequisite_send='d8d3a89cb3436100',state='Ready only after local admission; no send until prerequisite lands.',
        N_limits=[32,64,128],ring_stages=[2,3],producer_threads=128,consumer_threads=128,token_capacities=[32,128],
        bank='same-size u8 recoded scales; producer dequantizes each expert/channel/K64 tile once; no prepared dequant bank')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['payload'],manifest['sha256'],len(text))


if __name__=='__main__':main()
