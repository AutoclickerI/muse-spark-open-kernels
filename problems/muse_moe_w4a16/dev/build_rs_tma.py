"""K512 weight-strip bulk pipeline over the canonical V6 bank; one consumer WG."""
import ast
from pathlib import Path
p=Path(__file__).resolve().parent.parent
text=(p/'cand_rs128_7.py').read_text()
tree=ast.parse(text)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
s=ast.literal_eval(node.value)
helpers=r'''
__device__ __forceinline__ void cbar(){asm volatile("bar.sync 1, 128;":::"memory");}
__device__ __forceinline__ void mbinit(uint64_t* b,int n){asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;"::"r"(sptr(b)),"r"(n));}
__device__ __forceinline__ void mbwait(uint64_t* b,int phase){
 uint32_t done;
 do{asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p;}":"=r"(done):"r"(sptr(b)),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ void mbarrive(uint64_t* b){asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];"::"r"(sptr(b)):"memory");}
__device__ __forceinline__ void mbexpect(uint64_t* b,int n){asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(sptr(b)),"r"(n):"memory");}
__device__ __forceinline__ void bulk(void* d,const void* s,int n,uint64_t* b){
 asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(sptr(d)),"l"(s),"r"(n),"r"(sptr(b)):"memory");
}
'''
pos=s.index('template<int N,int MODE,int STAGES>')
s=s[:pos]+helpers+s[pos:]
s=s.replace('template<int N,int MODE,int STAGES>','template<int N,int MODE,int STAGES,int WS>')
s=s.replace('__launch_bounds__(128,2) grouped','__launch_bounds__(160,(N==128?1:2)) grouped')
a=s.index(' const int t=threadIdx.x,warp=t>>5,lane=t&31;')
b=s.index(' for(int item=blockIdx.x;',a)
s=s[:a]+r''' const int t=threadIdx.x,warp=t>>5,lane=t&31;
 constexpr int WBYTES=RB*4*4608;
 uint8_t* wshared=shared+STAGES*XBYTES;
 uint64_t* full=(uint64_t*)(wshared+WS*WBYTES);
 uint64_t* empty=full+WS;
 if(t==0){
  for(int i=0;i<WS;++i){mbinit(full+i,1);mbinit(empty+i,4);}
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
 }
 __syncthreads();
 const int total=tasks[32]*NFB;
 if(t>=128){
  if(t==128){
   int st=0,phase=0;
   for(int item=blockIdx.x;item<total;item+=gridDim.x){
    int ti=item/NFB,fb=item%NFB,e=0;
#pragma unroll
    for(int d=16;d;d>>=1)if(tasks[e+d]<=ti)e+=d;
    for(int k=0;k<8;++k){
     mbwait(empty+st,phase^1);
     mbexpect(full+st,WBYTES);
#pragma unroll
     for(int r=0;r<RB*4;++r){
      const uint8_t* source=weights+((size_t)e*MTILES+fb*4*RB+r)*TB;
      uint8_t* dest=wshared+st*WBYTES+r*4608;
      bulk(dest,source+k*4096,4096,full+st);
      bulk(dest+4096,source+WB+k*512,512,full+st);
     }
     if(++st==WS){st=0;phase^=1;}
    }
   }
  }
  return;
 }
 int wst=0,wphase=0;
''' +s[b:]
# The producer never participates in consumer-only barriers after initialization.
start=s.index(' int wst=0,wphase=0;')
head,body=s[:start],s[start:]
body=body.replace('__syncthreads();','cbar();')
body=body.replace('   uint32_t o[RB][16],sr[RB],su[RB];', '   if((kg&7)==0)mbwait(full+wst,wphase);\n   uint32_t o[RB][16],sr[RB],su[RB];')
body=body.replace('const uint8_t* wp=wt+rb*4*TB;', 'const uint8_t* wp=wshared+wst*WBYTES+(rb*4+warp)*4608;')
body=body.replace('wp+(kg*32+lane)*16', 'wp+((kg&7)*32+lane)*16')
body=body.replace('wp+WB+(kg/8)*512+lane*16+(kg%8)*2', 'wp+4096+lane*16+(kg&7)*2')
body=body.replace('   cbar();\n   if(kg+STAGES', '''   if((kg&7)==7){
    __syncwarp();if(lane==0)mbarrive(empty+wst);
    if(++wst==WS){wst=0;wphase^=1;}
   }
   cbar();
   if(kg+STAGES''')
s=head+body
# Replace the launch wrapper; configure shared-memory opt-in once in prepare.
a=s.index('void gemm(torch::Tensor')
s=s[:a]+r'''
void configure(){
 int dev,opt;cudaGetDevice(&dev);cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);
#define ATTR(N,M,W) C10_CUDA_CHECK(cudaFuncSetAttribute(canonical::grouped<N,M,2,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt))
#define BOTH(N) ATTR(N,0,1);ATTR(N,1,1);ATTR(N,0,2);ATTR(N,1,2)
 BOTH(8);BOTH(16);BOTH(32);BOTH(64);BOTH(128);
#undef BOTH
#undef ATTR
}
void gemm(torch::Tensor w,torch::Tensor gs,torch::Tensor x,torch::Tensor out,torch::Tensor pairs,torch::Tensor offsets,torch::Tensor tasks,int64_t mode,int64_t fixed_n,int64_t grid,int64_t stages,int64_t ws){
 TORCH_CHECK(stages==2 && (ws==1||ws==2));
 auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N,M,W) canonical::grouped<N,M,2,W><<<grid,160,N*128*2+W*36864+32,stream>>>(w.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)out.data_ptr(),pairs.data_ptr<int>(),offsets.data_ptr<int>(),tasks.data_ptr<int>()+(__builtin_ctz(N)-3)*33)
#define WIDTH(N,M,W) if(fixed_n==0||fixed_n==N){RUN(N,M,W);}
#define SET(M,W) WIDTH(8,M,W);WIDTH(16,M,W);WIDTH(32,M,W);WIDTH(64,M,W);WIDTH(128,M,W)
 if(mode){if(ws==2){SET(1,2);}else{SET(1,1);}}
 else{if(ws==2){SET(0,2);}else{SET(0,1);}}
#undef SET
#undef WIDTH
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
lines=text.splitlines(keepends=True);start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
text=text[:start]+'GROUPED_SRC = r"""'+s+'"""\n'+text[end:]
text=text.replace("'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);',", "'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\\n'\n                            'void configure();',")
text=text.replace("functions=['route', 'gemm']", "functions=['route', 'gemm', 'configure']")
text=text.replace("    state['grouped_mod'] = _grouped_module()\n", "    state['grouped_mod'] = _grouped_module()\n    state['grouped_mod'].configure()\n    state['weight_stages'] = 2 if torch.cuda.get_device_capability() == (9, 0) else 1\n")
text=text.replace('fixed_n, grid, stages)',"fixed_n, grid, stages, state['weight_stages'])")
(p/'cand_rs_tma.py').write_text(text)
