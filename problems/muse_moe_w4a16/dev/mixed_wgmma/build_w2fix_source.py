"""Derive the audited WGMMA mainloop with controlled W2 epilogues."""
from pathlib import Path
import re
from native_abi import fixed
H=Path(__file__).resolve().parent

def main():
 s=re.sub(r'\bft_', 'fx_', (H/'fused.cu').read_text()).replace('namespace mwft','namespace mwfx').replace('mwft::','mwfx::').replace('FT_CASES','FX_CASES')
 s=s.replace('int SPLIT,bool FIRST>','int SPLIT,bool FIRST,int MODE>')
 s=s.replace('int M,int cap,int start){','int M,int cap,int start,bool profiling){')
 s=s.replace('  int task=unit/SPLIT,','  uint64_t tick=profiling?clock64():0;uint64_t times[6]={};bool islast=false;\n  int task=unit/SPLIT,')
 s=s.replace('  if constexpr(SPLIT>1){','  if(profiling){uint64_t now=clock64();times[0]=now-tick;tick=now;}\n  if constexpr(SPLIT>1){')
 a=s.index('   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready')
 b=s.index('\n  }\n  __syncthreads();\n }\n}',a)
 old=s[a:b]
 old=old.replace('   __threadfence();__syncthreads();','   if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}\n   __threadfence();__syncthreads();',1)
 old=old.replace('   if(last){','   if(profiling){uint64_t now=clock64();times[2]=now-tick;tick=now;}\n   islast=last;\n   if(last){',1)
 old=old.replace('    if(t<NT){flags[t]=0;','    if(profiling){uint64_t now=clock64();times[3]=now-tick;tick=now;}\n    if(t<NT){flags[t]=0;',1)
 old=old.replace('    for(int r=w;r<NT;r+=4)if(flags[r])','    if(profiling){uint64_t now=clock64();times[4]=now-tick;tick=now;}\n    for(int r=w;r<NT;r+=4)if(flags[r])',1)
 c=old.index('     for(int c=l;c<4096;c+=32)');e=old.index('\n    }\n   }',c)
 combine=old[c:e]
 fast='''     int base[8];float alpha[8];
     #pragma unroll
     for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;base[j]=-1;alpha[j]=0;
      if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];base[j]=p;alpha[j]=INV[p]*TW[tok*8+j];}}
     for(int c=l*4;c<4096;c+=128){float4 sum=make_float4(0,0,0,0);
      #pragma unroll
      for(int j=0;j<8;++j)if(base[j]>=0){float4 z=*(const float4*)(Y+(size_t)base[j]*4096+c);float a=alpha[j];sum.x+=z.x*a;sum.y+=z.y*a;sum.z+=z.z*a;sum.w+=z.w*a;}
      __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);
      *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);
     }'''
 old=old[:c]+'     if constexpr(MODE==2){\n'+fast+'\n     }else{\n'+combine+'\n     }'+old[e:]
 s=s[:a]+'   if constexpr(MODE!=1){\n'+old+'\n   }else if(profiling){uint64_t now=clock64();times[1]=now-tick;tick=now;}'+s[b:]
 s=s.replace('  __syncthreads();\n }\n}\n#define FX_CASES','  __syncthreads();\n  if(profiling&&t==0){uint64_t*p=(uint64_t*)PART+(size_t)unit*8;\n   for(int i=0;i<5;++i)p[i]=times[i];p[5]=clock64()-tick;p[6]=islast;p[7]=clock64();}\n }\n}\n#define FX_CASES')
 a=s.index('#define FX_CASES');b=s.index('\n}',a)
 s=s[:a]+'''#define FX_CASES(F) \\
 F(0,16,128,4,1) F(1,32,128,4,1)'''+s[b:]
 a=s.index('int64_t fx_init()')
 s=s[:a]+'''int64_t fx_init(){int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
 #define INIT(ID,NT,KT,S,SK) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto set=[&]<bool F,int MODE>(){C10_CUDA_CHECK(cudaFuncSetAttribute(mwfx::fused<NT,KT,S,SK,F,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));};set.template operator()<true,1>();set.template operator()<false,1>();if constexpr(ID==0)set.template operator()<false,0>();}
 FX_CASES(INIT)
 #undef INIT
 return opt;
}
std::vector<int64_t> fx_info(int64_t which,int64_t first,int64_t mode){
 #define INFO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F,int MODE>(){cudaFuncAttributes a{};int blocks;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,mwfx::fused<NT,KT,S,SK,F,MODE>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,mwfx::fused<NT,KT,S,SK,F,MODE>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};if(first)return info.template operator()<true,1>();if(mode==1)return info.template operator()<false,1>();if constexpr(ID==0){TORCH_CHECK(mode==0);return info.template operator()<false,0>();}else TORCH_CHECK(false,"legacy control exists only forNT16");}
 switch(which){FX_CASES(INFO) default:TORCH_CHECK(false,"unknown fixed configuration");}
 #undef INFO
}
void fx_route(torch::Tensor ids,torch::Tensor c,torch::Tensor off,torch::Tensor order,torch::Tensor tasks,torch::Tensor slots,torch::Tensor pos,torch::Tensor state,torch::Tensor out,int64_t nt,int64_t start,int64_t clear_count){
 mwfx::route_init<<<128,256,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),c.data_ptr<int>(),off.data_ptr<int>(),order.data_ptr<int>(),tasks.data_ptr<int>(),slots.data_ptr<int>(),pos.data_ptr<int>(),state.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),ids.size(0),nt,start,clear_count);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void fx_run(std::vector<torch::Tensor> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t mode,int64_t profiling){
 TORCH_CHECK(v.size()==20);int M=v[11].size(0),cap=v[9].size(0);auto stream=at::cuda::getCurrentCUDAStream();
 #define GO(ID,NT,KT,S,SK) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\
 auto launch=[&]<bool FIRST,int MODE>(){mwfx::fused<NT,KT,S,SK,FIRST,MODE><<<grid,128,bytes,stream>>>(v[0].data_ptr<int>(),v[1].data_ptr<uint8_t>(),v[2].data_ptr<float>(),(const __nv_bfloat16*)v[3].data_ptr(),v[4].data_ptr<int>(),v[5].data_ptr<int>(),v[6].data_ptr<int>(),v[7].data_ptr<int>(),v[8].data_ptr<int>(),v[11].data_ptr<int>(),v[12].data_ptr<float>(),v[13].data_ptr<int>(),(const __nv_bfloat16*)v[14].data_ptr(),v[10].data_ptr<float>(),(__nv_bfloat16*)v[9].data_ptr(),v[15].data_ptr<float>(),v[16].data_ptr<float>(),v[17].data_ptr<float>(),v[18].data_ptr<int>(),(__nv_bfloat16*)v[19].data_ptr(),M,cap,start,profiling);};if(first)launch.template operator()<true,1>();else if(mode==1)launch.template operator()<false,1>();else{TORCH_CHECK(mode==0&&which==0);if constexpr(ID==0)launch.template operator()<false,0>();}break;}
 switch(which){FX_CASES(GO) default:TORCH_CHECK(false,"unknown fixed configuration");}
 #undef GO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
 tail=(H/'tail_probe.cu').read_text();a=tail.index('__global__ void finish(');b=tail.index('\n}\nvoid tp_run',a)
 s+='namespace mwfx {\n'+tail[a:b]+'\n}\n'
 s+='''void fx_finish(std::vector<torch::Tensor>v,int64_t start){mwfx::finish<<<v[11].size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(v[5].data_ptr<int>(),v[11].data_ptr<int>(),v[12].data_ptr<float>(),v[13].data_ptr<int>(),v[15].data_ptr<float>(),v[16].data_ptr<float>(),(__nv_bfloat16*)v[19].data_ptr(),v[11].size(0),start);C10_CUDA_KERNEL_LAUNCH_CHECK();}\n'''
 (H/'w2fix.cu').write_text(fixed(s))
 print('w2fix.cu',len(s))

if __name__=='__main__':main()
