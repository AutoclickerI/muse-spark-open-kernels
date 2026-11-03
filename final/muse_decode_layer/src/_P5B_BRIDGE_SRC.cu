#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstring>
#include <stdexcept>
/*
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;
    index_t v_dim_stride;

    // The number of heads.
    int h, h_k;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // For FP8 scaling
    float * __restrict__ q_descale_ptr;
    float * __restrict__ k_descale_ptr;
    float * __restrict__ v_descale_ptr;
    index_t q_descale_batch_stride;
    index_t q_descale_head_stride;
    index_t k_descale_batch_stride;
    index_t k_descale_head_stride;
    index_t v_descale_batch_stride;
    index_t v_descale_head_stride;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim;
    int total_q, total_k, total_knew;
    int b_k;  // When having KV cache and with cache_batch_idx, K & V might have larger batch size than Q
    int dv, dv_rounded;  // For the case where V headdim is different from Q/K headdim

    // The scaling factors for the kernel.
    float scale_softmax;
    float softcap;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ cu_seqlens_knew;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each q/k sequence.
    int *__restrict__ seqused_q;
    int *__restrict__ seqused_k;

    // The stride between rows of Oaccum.
    index_t oaccum_split_stride;
    index_t oaccum_batch_stride;
    index_t oaccum_row_stride;
    index_t oaccum_head_stride;

    // The stride between rows of LSEaccum.
    index_t lseaccum_split_stride;
    index_t lseaccum_batch_stride;
    index_t lseaccum_head_stride;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    void *__restrict__ qv_ptr;
    index_t qv_batch_stride;
    index_t qv_row_stride;
    index_t qv_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;
    int *__restrict__ seqlens_rotary;

    // The indices to index into the KV cache.
    int * __restrict__ kv_batch_idx;

    // Paged KV cache
    int * __restrict__ page_table;
    index_t page_table_batch_stride;
    int page_size;
    int num_pages;
    bool pagedkv_tma;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    int attention_chunk;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_fp32;
    bool is_e4m3;
    bool is_causal;
    bool is_local;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version
    bool pack_gqa;

    int * __restrict__ tile_count_semaphore;
    int * __restrict__ num_m_blocks_ptr;
    // int * __restrict__ num_n_blocks_ptr;
    int * __restrict__ num_splits_dynamic_ptr;
    int * __restrict__ varlen_batch_idx_ptr; // virtual -> actual
    int * __restrict__ num_nheads_in_l2_ptr;
    bool skip_scheduler_metadata_computation;
    bool varlen_sort_batches;
    int tile_count_semaphore_offset;
    bool head_swizzle;
    bool prepare_varlen_pdl;

    int arch;
    int num_sm;
    bool only_qv;
    
    // Learnable sink
    void *__restrict__ sink_ptr;
    int * __restrict__ sparse_mask_fine;

    // sparse mask
    int sparse_mask_k_block_size;
    int sparse_mask_q_tile_size;
    int sparse_mask_max_k_blocks;

    int sparse_mask_fine_k_stride;
    int sparse_mask_fine_q_stride;

    bool use_sparse_mask;
};

////////////////////////////////////////////////////////////////////////////////////////////////////


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
