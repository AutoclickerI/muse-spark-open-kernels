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
    cpp=CPP_PRELUDE+'int64_t pc_init();std::vector<int64_t> pc_info(int64_t,int64_t,int64_t);void pc_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void pc_meta(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void pc_probe(std::vector<int64_t>,int64_t);\n'
    cuda+='''
__global__ void pc_meta_kernel(const int*C,const int*ORDER,int*T,int*T32,int*MID,float*SS,int cap,int cut,int pcut){
 int t=threadIdx.x,all=blockIdx.x*blockDim.x+t;
 for(int i=all;i<cap*64;i+=gridDim.x*blockDim.x)SS[i]=0.f;
 if(blockIdx.x==0&&t<32){int n=C[ORDER[t]],v=n>pcut,b=n>cut?(n+31)/32:0,m=(n>cut&&n<=pcut)?(n+31)/32:0;
  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),c=__shfl_up_sync(~0u,b,d),e=__shfl_up_sync(~0u,m,d);if(t>=d){v+=a;b+=c;m+=e;}}
  T[t+1]=v;T32[t+1]=b;MID[t+1]=m;if(t==0){T[0]=0;T32[0]=0;MID[0]=0;}}
}
void pc_meta(std::vector<int64_t>v,int64_t cap,int64_t cut,int64_t pcut,int64_t stream){pc_meta_kernel<<<128,256,0,(cudaStream_t)stream>>>((int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(float*)v[5],cap,cut,pcut);WX_CUDA_CHECK(cudaGetLastError());}
__global__ void pc_decode_probe(const int*Q,const uint8_t*S,__nv_bfloat16*OUT){
 __shared__ __align__(128) unsigned char raw[2304];int t=threadIdx.x,w=t>>5,l=t&31;
 for(int g=0;g<2;++g){int z=w*128+g;
  if(l<16)pc::async_copy(raw+(g*128+w*32)*8+l*16,Q+z*64+l*4,16,1);
  if(l<8)pc::async_copy(raw+2048+g*128+w*32+l*4,S+z*32+l*4,4,1);}
 asm volatile("cp.async.commit_group;cp.async.wait_group 0;":::"memory");__syncwarp();unsigned af[16];
 for(int g=0;g<2;++g){uint2 v=*(uint2*)(raw+(g*128+t)*8);unsigned sf=*(unsigned*)(raw+2048+g*128+w*32+(l>>2)*4);pc::dq(v.x,v.y,sf,af+g*8);}
 for(int atom=0;atom<4;++atom)for(int j=0;j<4;++j){int r=w*16+(l>>2)+(j&1)*8,k=atom*16+(l&3)*2+(j>>1)*8;*(unsigned*)(OUT+r*64+k)=af[atom*4+j];}
}
void pc_probe(std::vector<int64_t>v,int64_t stream){pc_decode_probe<<<1,128,0,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(__nv_bfloat16*)v[2]);WX_CUDA_CHECK(cudaGetLastError());}
'''
    (HERE/'pc.cu').write_text(cuda)
    helpers=subset((ROOT/'prof/lab/moe_m128_phase_mix.py').read_text(),('dc_weights','dc_fixtures','dc_oracles','MWGraph','pm_error'))
    hh=(BASE/'hh_cell.py').read_text();assert hashlib.sha256(hh.encode()).hexdigest()=='365b623992367a1d45d3dae06a2096ddd0e6879deffab2e3924678a47997b621'
    small=(ROOT/'problems/muse_decode_layer/dev/nt1_branches/nt1_lop3_cell.py').read_text()
    text=helpers+'\n'+hh+'\n'+small+'\nimport gc\nimport time\nimport statistics\n'
    text+='PC_CUDA='+repr(cuda)+'\nPC_CPP='+repr(cpp)+'\n'+(HERE/'driver.py.inc').read_text()
    assert not rejected_words(text),rejected_words(text)
    out=ROOT/'prof/lab/moe_m128_producer_consumer.py';out.write_text(text)
    manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        control_HH='u8 hh_cell (S7 idist; bit-identical to 770809cf u16 H_H)',H_H_sha256=hashlib.sha256(hh.encode()).hexdigest(),small_cell_sha256=hashlib.sha256(small.encode()).hexdigest(),
        prerequisite_send='d8d3a89cb3436100',state='Ready only after local admission; no send until prerequisite lands.',
        N_limits=[32,64,128],ring_stages=[2,3],producer_threads=128,consumer_threads=128,token_capacities=[32,128],
        bank='same-size u8 recoded scales; producer dequantizes each expert/channel/K64 tile once; no prepared dequant bank')
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['payload'],manifest['sha256'],len(text))


if __name__=='__main__':main()
