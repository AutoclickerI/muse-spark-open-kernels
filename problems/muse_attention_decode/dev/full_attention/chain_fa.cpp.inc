#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
// PARAM_HEADER
static_assert(sizeof(Flash_fwd_params)==728,"FA3 ABI size");
static_assert(offsetof(Flash_fwd_params,arch)==0x2a4,"FA3 ABI arch");
static_assert(offsetof(Flash_fwd_params,num_splits)==0x268,"FA3 ABI split");
using MainFn=void(*)(Flash_fwd_params&,cudaStream_t);
using CombineFn=void(*)(Flash_fwd_params&,cudaStream_t,bool);
using SetFn=void(*)(Flash_fwd_params&,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,size_t,at::Tensor,at::Tensor,at::Tensor,at::Tensor,void*,void*,void*,void*,void*,float,float,int,int,int,float,int);
static MainFn main_fn=nullptr;static CombineFn combine_fn=nullptr;static SetFn set_fn=nullptr;
void fa_bind(std::string path){
 void*h=dlopen(path.c_str(),RTLD_NOW|RTLD_LOCAL);if(!h)throw std::runtime_error(dlerror());
 main_fn=(MainFn)dlsym(h,"_Z11run_mha_fwdR16Flash_fwd_paramsP11CUstream_st");
 combine_fn=(CombineFn)dlsym(h,"_Z19run_mha_fwd_combineR16Flash_fwd_paramsP11CUstream_stb");
 set_fn=(SetFn)dlsym(h,"_Z16set_params_fpropR16Flash_fwd_paramsmmmmmmmmmN2at6TensorES2_S2_S2_PvS3_S3_S3_S3_ffiiifi");
 if(!main_fn||!combine_fn||!set_fn)throw std::runtime_error("FA3 launcher export missing");
}
Flash_fwd_params params(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor o,at::Tensor lse,at::Tensor oa,at::Tensor la,at::Tensor seq,at::Tensor cu,at::Tensor pt,at::Tensor meta,at::Tensor ks,at::Tensor vs,int L,int window,int split,float scale){
 if(!set_fn)throw std::runtime_error("bind FA3 in prepare first");int B=seq.numel(),HQ=q.size(-2),HK=k.size(-2),ps=k.size(1),pages=pt.size(1);window=window<pages*ps?window:0;
 // Guard both header size and library writes; set_params also initializes all
 // optional sink/sparse/QV fields. No private module is modified.
 struct alignas(16) Buf{Flash_fwd_params p;unsigned char guard[32];}buf;std::memset(&buf,0xa5,sizeof(buf));
 set_fn(buf.p,B,L,pages*ps,(L+127)/128*128,(pages*ps+127)/128*128,HQ,HK,64,64,q,k,v,o,cu.data_ptr(),nullptr,nullptr,seq.data_ptr(),lse.data_ptr(),0.f,scale,window?window-1:-1,0,0,0.f,0);
 for(auto x:buf.guard)if(x!=0xa5)throw std::runtime_error("FA3 ABI overwrite");auto&p=buf.p;
 TORCH_CHECK(p.b==B&&p.h==HQ&&p.h_k==HK&&p.seqlen_q==L&&p.d==64&&p.q_ptr==q.data_ptr()&&p.is_e4m3,"FA3 ABI fields mismatch");
 p.b_k=B;p.dv=p.dv_rounded=64;p.total_q=B*L;p.total_k=B*ps;p.total_knew=0;
 p.page_table=(int*)pt.data_ptr();p.page_table_batch_stride=pt.stride(0);p.page_size=ps;p.num_pages=k.size(0);p.pagedkv_tma=false;
 p.num_splits=split;p.pack_gqa=true;p.skip_scheduler_metadata_computation=true;p.varlen_sort_batches=!window;p.head_swizzle=true;p.prepare_varlen_pdl=true;
 int br=(B+3)/4*4;TORCH_CHECK(meta.numel()==(window?3:4)*br+1,"FA3 shape schedule layout");int*mt=(int*)meta.data_ptr();p.num_splits_dynamic_ptr=mt;p.num_m_blocks_ptr=mt+br;p.varlen_batch_idx_ptr=window?nullptr:mt+2*br;p.num_nheads_in_l2_ptr=mt+(window?2:3)*br;p.tile_count_semaphore_offset=(window?3:4)*br;p.tile_count_semaphore=mt+p.tile_count_semaphore_offset;
 p.oaccum_ptr=oa.data_ptr();p.softmax_lseaccum_ptr=la.data_ptr();p.oaccum_split_stride=oa.stride(0);p.oaccum_row_stride=oa.stride(-2);p.oaccum_head_stride=oa.stride(-3);p.oaccum_batch_stride=0;
 p.lseaccum_split_stride=la.stride(0);p.lseaccum_head_stride=la.stride(-2);p.lseaccum_batch_stride=0;
 p.k_descale_ptr=(float*)ks.data_ptr();p.v_descale_ptr=(float*)vs.data_ptr();p.k_descale_batch_stride=p.k_descale_head_stride=p.v_descale_batch_stride=p.v_descale_head_stride=0;
 return p;
}
std::vector<int64_t> fa_probe(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor o,at::Tensor lse,at::Tensor oa,at::Tensor la,at::Tensor seq,at::Tensor cu,at::Tensor pt,at::Tensor meta,at::Tensor ks,at::Tensor vs,int64_t L,int64_t window,int64_t split,double scale){auto p=params(q,k,v,o,lse,oa,la,seq,cu,pt,meta,ks,vs,L,window,split,scale);return {sizeof(p),p.arch,p.num_sm,p.num_splits,p.is_e4m3,p.is_local,p.is_causal};}
void fa_main(at::Tensor q,at::Tensor k,at::Tensor v,at::Tensor o,at::Tensor lse,at::Tensor oa,at::Tensor la,at::Tensor seq,at::Tensor cu,at::Tensor pt,at::Tensor meta,at::Tensor ks,at::Tensor vs,int64_t L,int64_t window,int64_t split,double scale,int64_t combine,int64_t stream){
 auto p=params(q,k,v,o,lse,oa,la,seq,cu,pt,meta,ks,vs,L,window,split,scale);auto st=(cudaStream_t)stream;
 if(combine&2){p.k_descale_ptr=nullptr;p.v_descale_ptr=nullptr;}
 main_fn(p,st);
 if((combine&1)&&split>1){p.is_bf16=true;combine_fn(p,st,true);}
 // The native parallel reducer resets the scheduler semaphore when it replaces
 // FA3's combine. It reads every partial and applies RMSNorm/gate in one launch.
}
