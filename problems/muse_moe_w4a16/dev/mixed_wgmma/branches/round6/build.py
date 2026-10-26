"""Repair both-consumer X retirement, then test fixed64/S3/S4 with2CTA targets."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;B=H.parent;sys.path.insert(0,str(B))
spec=importlib.util.spec_from_file_location('p3b_r3_repair_builder',B/'round3/build.py');r3=importlib.util.module_from_spec(spec);spec.loader.exec_module(r3)
from light_build import value
def fun(src,name):return ast.get_source_segment(src,[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name==name][-1])
def join_consumers(cu):
 old='if constexpr(S>1)if(sub==0){if(k+S-1<32)';assert cu.count(old)==1
 cu=cu.replace(old,'if constexpr(S>1)if(sub==0){__syncthreads();if(k+S-1<32)')
 old='if constexpr(S==1)if(k+1<32)issue(k+1,(st+1)%S);';assert cu.count(old)==1
 cu=cu.replace(old,'if constexpr(S==1){__syncthreads();if(k+1<32)issue(k+1,(st+1)%S);}')
 old='for(int b=0;b<4;++b)mm<NT>(acc,af[sub]+b*4,bs+st*NT*KT+sub*NT*64,b,desc0+(st*NT*KT+sub*NT*64)/8);'
 assert cu.count(old)==1
 args='acc,af[sub]+b*4,bs+st*NT*KT+sub*NT*64,b,desc0+(st*NT*KT+sub*NT*64)/8'
 return cu.replace(old,'for(int b=0;b<4;++b){if constexpr(NT==16)mwbr::mma<16>('+args+');else mm<NT>('+args+');}')
def chunk_cuda(deep):
 a=deep.index('namespace oc {');z=deep.index('template<int MAX,bool F,int CS>__global__',a)
 dev=deep[a:z].replace('namespace oc {','namespace chunk {',1)
 kernel=r'''
template<int DEPTH,bool F>__global__ void __launch_bounds__(256,2) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){
 constexpr int S=BR_HOPPER?DEPTH:3,MT=F?64:32;
 extern __shared__ __align__(1024) unsigned char ocsm[];float*red=(float*)ocsm+(threadIdx.x>>7)*256;
 for(int u=blockIdx.x;u<32*MT;u+=gridDim.x){int e=ORDER[u/MT],mt=u%MT,n=C[e];if(n<=16)continue;
  for(int row=0;row<n;row+=64){int rem=min(n-row,64);
#define CH_TILE(NT) tile<NT,F,1,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt*2+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red)
   if(rem<=32){CH_TILE(32);}else{CH_TILE(64);}
#undef CH_TILE
  }
 }
}
}
template<int S,bool F>std::vector<int64_t> ch_resource(){int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));int bytes=major==9?oc::dyn<S,64>():oc::dyn<3,64>();cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncSetAttribute(chunk::kernel<S,F>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));BR_CUDA_CHECK(cudaFuncGetAttributes(&a,chunk::kernel<S,F>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,chunk::kernel<S,F>,256,bytes));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,n};}
std::vector<int64_t> ch_info(int64_t s,int64_t f){if(s==3)return f?ch_resource<3,true>():ch_resource<3,false>();if(s==4)return f?ch_resource<4,true>():ch_resource<4,false>();BR_REQUIRE(false);return {};}
void ch_run(std::vector<int64_t>v,int64_t first,int64_t depth,int64_t grid,int64_t M,int64_t stream){int dev,major;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,dev));int bytes=major==9?(depth==3?oc::dyn<3,64>():oc::dyn<4,64>()):oc::dyn<3,64>();
#define CH_GO(S,F) chunk::kernel<S,F><<<grid,256,bytes,(cudaStream_t)stream>>>(BR_ARGS(v),(int)M)
 if(depth==3){if(first){CH_GO(3,true);}else{CH_GO(3,false);}}else if(depth==4){if(first){CH_GO(4,true);}else{CH_GO(4,false);}}else BR_REQUIRE(false);
#undef CH_GO
 BR_CUDA_CHECK(cudaGetLastError());}
'''
 return dev+kernel
def main():
 r3.main();src=(B/'round3/tail_payload.py').read_text();deep=join_consumers((B/'round2/cta2s3.cu').read_text())
 cu=r3.cuda(deep,deep)+'\n'+chunk_cuda(deep);cpp=value(src,'TAIL_CPP')+'void ch_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);std::vector<int64_t> ch_info(int64_t,int64_t);'
 plan=fun(src,'nf_light_plan').replace('def nf_light_plan(','def chunk_plan(').replace('streams, light_grid):','streams, light_grid, depth):')
 plan=plan.replace('mod.br_run(p13, 26, grid, m, side.cuda_stream)','mod.ch_run(p13, 1, depth, grid, m, side.cuda_stream)').replace('mod.br_run(p2, 21, grid, m, side.cuda_stream)','mod.ch_run(p2, 0, depth, grid, m, side.cuda_stream)').replace('grid = 1056 if m <= 128 else 2112','grid = sms * 4')
 assert 'depth):' in plan and 'ch_run' in plan
 body=src[:src.index('TAIL_CUDA=')]+'\nSAFE_CUDA='+repr(cu)+'\nSAFE_CPP='+repr(cpp)+'\nNF_OPTS={}\n'
 body+='\n'.join(fun(src,n) for n in ('reuse_plan','tail_plan','nf_module','op_call','op_error','op_pair'))+'\n'+plan+'\n'+(H/'runtime.py.inc').read_text()
 compile(body,'<shared-X repair>','exec');(H/'p3b_repair_payload.py').write_text(body);(H/'repair.cu').write_text(cu)
 for arm,name in enumerate(('s14','cta2s3','static','queue','queue_s3','chunk_s3','chunk_s4')):
  if not arm:continue
  adapter='# okc: problem=muse_moe_w4a16\ndef lab_cold_l2():pass\n'+body+f'\nSCREEN_ARM={arm}\n'+'''def prepare(weights):
 op_build()
 w={k:(v[:32].contiguous() if k.startswith(('w13','w2')) else v) for k,v in weights.items()};w['post_expert_norm']=w['post_norm_w'];return op_prepare(w)
def kernel_fn(data,state):return op_call(data,state,SCREEN_ARM)
'''
  (H/f'screen_{name}.py').write_text(adapter)
 out=dict(payload_sha256=hashlib.sha256(body.encode()).hexdigest(),cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),base='S14 f1c77ce6',repair='Join both WGs before shared X reuse; route NT16 to the existing native BF16 WGMMA helper.',arms=['S14','SAFE_CTA2S3','SAFE_STATIC','SAFE_QUEUE','SAFE_QUEUE_S3','CHUNK64_S3_2CTA','CHUNK64_S4_2CTA'])
 # Keep the host baseline reproducible without retaining a duplicate kernel file.
 host_snapshot=B.parents[4]/'runs/muse_moe_w4a16/20261026T010122Z_submit_dd539517bc8344dd.py'
 host=host_snapshot.read_text().split('# ---- h200_lab wrapper:',1)[0].rstrip()+'\n'
 assert hashlib.sha256(host.encode()).hexdigest()=='84eb293228a3e3132a73cd5ee3f6ac1d252abe9b2384a70090e1540239a09e7e'
 (H/'compact_host.py').write_text(host)
 out['host_base_sha256']=hashlib.sha256(host.encode()).hexdigest()
 out['host_base_provenance']=str(host_snapshot.relative_to(B.parents[4]))
 (H/'manifest.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':main()
