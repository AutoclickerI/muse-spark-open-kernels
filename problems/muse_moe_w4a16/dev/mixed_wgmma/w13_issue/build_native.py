"""W13-only issue cuts: preserve shared provenance, then vectorize Q/SF transfers."""
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent

def main():
 s=(P/'recode/recode.cu').read_text();s=s[:s.index('#define RC_CASES')]+ '\n}\n'
 a=s.index('__global__ void route_init');b=s.index('template<int NT,int KT,int S,int SPLIT,bool FIRST',a);s=s[:a]+s[b:]
 s=s.replace('namespace mwrc','namespace mwwi').replace('int MODE,bool RECODED>','int MODE,bool RECODED,bool VEC,int FILTER=0,bool QUAD=false>')
 s=s.replace('unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;', 'unsigned w;asm(\"lop3.b32 %0,%1,0x60306030,%2,0xe2;\":\"=r\"(w):\"r\"(a),\"r\"(b));unsigned mask=0x81c081c0u;')
 s=s.replace('template<int NT,int KT>__device__ __forceinline__ void load_stage(','template<int NT,int KT,bool VEC>__device__ __forceinline__ void load_stage(')
 a=s.index(' int t=threadIdx.x,w=t>>5,l=t&31;',s.index('void load_stage'));b=s.index('\n #pragma unroll\n for(int c=0;',a)
 old=s[a:b]
 new=''' int t=threadIdx.x,w=t>>5,l=t&31;
 if constexpr(VEC){
  // Q: two warps issue16B, replacing four warps'8B copies. Keep .ca policy.
  if(t<64){unsigned group=t>>4,lane=t&15;
   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;
   #pragma unroll
   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*src=Q+base+g*64;
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");}
  }
  // SF: one warp issues all512B once, replacing16 quarter-warp scalar copies.
  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;
   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;
   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*src=S+srcidx;
   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");
  }
 }else{
'''+old[old.index(' #pragma unroll'):]+ '\n }'
 s=s[:a]+new+s[b:]
 s=s.replace('      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);', '      if constexpr(QUAD){unsigned sc=*(const unsigned char*)(ss+st*SB+z*128+t);unsigned sbits;\n       if constexpr(RECODED)sbits=sc*0x00100010u+0x70007000u;else sbits=lut[sc];\n       dq_scaled(q.x,q.y,__shfl_sync(~0u,sbits,0,4),__shfl_sync(~0u,sbits,1,4),__shfl_sync(~0u,sbits,2,4),__shfl_sync(~0u,sbits,3,4),af[reg]+g*8);\n      }else{unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq<RECODED>(q.x,q.y,sf,lut,af[reg]+g*8);}')
 helper='''__device__ __forceinline__ void dq_scaled(unsigned a,unsigned b,unsigned s0,unsigned s1,unsigned s2,unsigned s3,unsigned*out){
 unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;
 out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);
}
'''
 s=s.replace('template<bool RECODED>__device__',helper+'template<bool RECODED>__device__',1)
 s=s.replace('load_stage<NT,KT>','load_stage<NT,KT,VEC>')
 s=s.replace('int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];', 'int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  if constexpr(FILTER==1){if(count>16)continue;}\n  if constexpr(FILTER==2){if(count<=16)continue;}')
 old=' extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);'
 assert old in s;s=s.replace(old,' extern __shared__ __align__(1024) unsigned char typed_sm[];\n unsigned char*sm=typed_sm;')
 s+='''
int64_t wi_init(){
 constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;
 auto set=[&]<bool R,bool V>(){WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,R,V>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<false,false>();set.template operator()<false,true>();set.template operator()<true,true>();
 WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,false,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,0,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
 constexpr int cold_bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*16*128*2;WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<16,128,4,1,true,1,true,true,1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,cold_bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(mwwi::fused<32,128,4,1,true,1,true,true,2>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));return bytes;
}
std::vector<int64_t> wi_info(int64_t policy){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;
 auto info=[&]<bool R,bool V>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<32,128,4,1,true,1,R,V>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwwi::fused<32,128,4,1,true,1,R,V>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(policy==0)return info.template operator()<false,false>();if(policy==1)return info.template operator()<false,true>();WX_CHECK(policy==2);return info.template operator()<true,true>();}
void wi_run(std::vector<int64_t>v,int64_t policy,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){
 constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*32*128*2;
 auto launch=[&]<bool R,bool V>(){mwwi::fused<32,128,4,1,true,1,R,V><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(policy==0)launch.template operator()<false,false>();else if(policy==1)launch.template operator()<false,true>();else{WX_CHECK(policy==2);launch.template operator()<true,true>();}WX_CUDA_CHECK(cudaGetLastError());}
'''
 s+='''
void wi_mixed(std::vector<int64_t>v,int64_t hot,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){
 auto launch=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,true,true,FILTER><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(hot)launch.template operator()<32,2>();else launch.template operator()<16,1>();WX_CUDA_CHECK(cudaGetLastError());}
'''
 s+='''
std::vector<int64_t> wi_mixed_info(int64_t hot){auto info=[&]<int NT,int FILTER>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;cudaFuncAttributes a{};int n;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwwi::fused<NT,128,4,1,true,1,true,true,FILTER>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};};return hot?info.template operator()<32,2>():info.template operator()<16,1>();}
'''
 s+='''
void wi_quad(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t cap,int64_t start,int64_t stream){
 auto launch=[&]<int NT,bool R,int F>(){constexpr int bytes=1023+((4*(64*128/2+64*128/16)+1023)&~1023)+4*NT*128*2;mwwi::fused<NT,128,4,1,true,1,R,true,F,true><<<grid,128,bytes,(cudaStream_t)stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(__nv_bfloat16*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(float*)v[10],(__nv_bfloat16*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)v[19],M,cap,start,false);};if(kind==0)launch.template operator()<32,false,0>();else if(kind==1)launch.template operator()<32,true,0>();else if(kind==2)launch.template operator()<16,true,1>();else{WX_CHECK(kind==3);launch.template operator()<32,true,2>();}WX_CUDA_CHECK(cudaGetLastError());}
'''
 (H/'issue.cu').write_text(s);print('issue.cu',len(s))

if __name__=='__main__':main()
