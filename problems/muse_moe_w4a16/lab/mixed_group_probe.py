"""CUTLASS SM90 pointer-array grouped mixed-input FP4 probe."""
import base64 as _g_b64
import zlib as _g_zlib
import torch as _g_torch
from torch.utils.cpp_extension import load_inline as _g_load_inline
_G_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
using namespace cute;
using EA = cutlass::bfloat16_t;
using EB = cute::tuple<cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
using TS = cute::Shape<cute::_128,cute::_128,cute::_64>;
using CS = cute::Shape<cute::_1,cute::_1,cute::_1>;
using ML = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    EA, cutlass::layout::RowMajor*, 8,
    EB, cutlass::layout::ColumnMajor*, 32,
    float, TS, CS, cutlass::gemm::collective::StageCountAuto, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using EL = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TS, CS, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::PtrArrayNoSmemWarpSpecialized,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t,float>>::CollectiveOp;
using PS = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
using GK = cutlass::gemm::kernel::GemmUniversal<PS,ML,EL,void>;
using GA = cutlass::gemm::device::GemmUniversalAdapter<GK>;


#include <memory>
#include <vector>
#include <cstring>
struct GroupCtx {
  GA op;
  int N=0,K=0,G=0;
  torch::Tensor pa,pb,ps,pd,shdev,shhost,work;
  std::vector<torch::Tensor> aa,bb,ss,dd;
  std::vector<int64_t> ha,hb,hs,hd;
  std::vector<cute::Stride<int64_t,cute::Int<1>,cute::Int<0>>> sta,stb;
  std::vector<cute::Stride<cute::Int<1>,int64_t,int64_t>> sts;
  std::vector<cute::Shape<int,int,int>> hshape;
  PS pshape;
  bool ready=false;
};
static std::unique_ptr<GroupCtx> gctx;
void run_group(std::vector<torch::Tensor> aa, std::vector<torch::Tensor> bb,
               std::vector<torch::Tensor> ss, std::vector<torch::Tensor> dd) {
  TORCH_CHECK(aa.size()==bb.size() && aa.size()==ss.size() && aa.size()==dd.size());
  int G=(int)aa.size(); TORCH_CHECK(G>0);
  int K=(int)aa[0].size(1), N=(int)dd[0].size(1);
  auto dev=aa[0].device(); auto stream=at::cuda::getCurrentCUDAStream();
  if (!gctx || !gctx->ready || gctx->N!=N || gctx->K!=K || gctx->G!=G) {
    gctx=std::make_unique<GroupCtx>(); auto& c=*gctx; c.N=N;c.K=K;c.G=G;c.aa=aa;c.bb=bb;c.ss=ss;c.dd=dd;
    auto opts_i=torch::TensorOptions().dtype(torch::kInt64).device(dev);
    c.pa=torch::empty({G},opts_i);c.pb=torch::empty({G},opts_i);c.ps=torch::empty({G},opts_i);c.pd=torch::empty({G},opts_i);
    c.ha.resize(G);c.hb.resize(G);c.hs.resize(G);c.hd.resize(G);c.hshape.resize(G);c.sta.resize(G);c.stb.resize(G);c.sts.resize(G);
    for(int i=0;i<G;++i){c.ha[i]=(int64_t)aa[i].data_ptr();c.hb[i]=(int64_t)bb[i].data_ptr();c.hs[i]=(int64_t)ss[i].data_ptr();c.hd[i]=(int64_t)dd[i].data_ptr();c.hshape[i]=cute::Shape<int,int,int>{(int)aa[i].size(0),N,K};c.sta[i]={int64_t(K),cute::Int<1>{},cute::Int<0>{}};c.stb[i]={int64_t(K),cute::Int<1>{},cute::Int<0>{}};c.sts[i]={cute::Int<1>{},int64_t(N),int64_t(0)};}
    c.shdev=torch::empty({G,3},torch::TensorOptions().dtype(torch::kInt32).device(dev));c.shhost=torch::empty({G,3},torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    std::memcpy(c.shhost.data_ptr(),c.hshape.data(),G*sizeof(cute::Shape<int,int,int>));
    C10_CUDA_CHECK(cudaMemcpyAsync(c.pa.data_ptr(),c.ha.data(),G*sizeof(int64_t),cudaMemcpyHostToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(c.pb.data_ptr(),c.hb.data(),G*sizeof(int64_t),cudaMemcpyHostToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(c.ps.data_ptr(),c.hs.data(),G*sizeof(int64_t),cudaMemcpyHostToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(c.pd.data_ptr(),c.hd.data(),G*sizeof(int64_t),cudaMemcpyHostToDevice,stream));C10_CUDA_CHECK(cudaMemcpyAsync(c.shdev.data_ptr(),c.shhost.data_ptr(),G*3*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    c.pshape.num_groups=G;c.pshape.problem_shapes=reinterpret_cast<Shape<int,int,int>*>(c.shdev.data_ptr());c.pshape.host_problem_shapes=reinterpret_cast<const Shape<int,int,int>*>(c.shhost.data_ptr());
    ML::Arguments ma{};ma.ptr_A=reinterpret_cast<const EA**>(c.pa.data_ptr());ma.dA=c.sta.data();ma.ptr_B=reinterpret_cast<const cutlass::float_e2m1_t**>(c.pb.data_ptr());ma.dB=c.stb.data();ma.ptr_S=reinterpret_cast<const cutlass::float_e4m3_t**>(c.ps.data_ptr());ma.dS=c.sts.data();ma.chunk_size=16;
    typename EL::Arguments ea{};ea.thread=typename EL::ThreadEpilogueOp::Params(1.f,0.f);ea.ptr_C=nullptr;ea.dC={};ea.ptr_D=reinterpret_cast<cutlass::bfloat16_t**>(c.pd.data_ptr());ea.dD={int64_t(N),cute::Int<1>{},int64_t(0)};
    typename GA::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,c.pshape,ma,ea};auto ws=GA::get_workspace_size(args);c.work=torch::empty({(long)ws},torch::TensorOptions().dtype(torch::kUInt8).device(dev));auto st=c.op.initialize(args,c.work.numel()?c.work.data_ptr():nullptr,stream);TORCH_CHECK(st==cutlass::Status::kSuccess,"init ",int(st));c.ready=true;
  }
  auto st=gctx->op.run(stream); TORCH_CHECK(st==cutlass::Status::kSuccess,"run ",int(st));
}
"""
_G_MOD = None
def _g_module():
    global _G_MOD
    if _G_MOD is None:
        _G_MOD = _g_load_inline(name='muse_cutlass_grouped_mixed_84b94286', cpp_sources='void run_group(std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>);', cuda_sources=_G_CUDA, functions=['run_group'], extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/deep_gemm/include'], extra_cflags=['-std=c++20','-O3'], extra_cuda_cflags=['-std=c++20','-O3','-gencode=arch=compute_90a,code=sm_90a'], verbose=False)
    return _G_MOD
def _g_ref(x,wq,sf,gs):
    grid=_g_torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    c=_g_torch.stack((wq&15,wq>>4),-1).reshape(wq.shape[0],-1);mag=grid[(c&7).long()];v=_g_torch.where((c&8)!=0,-mag,mag)
    return x.float() @ (v*sf.float().repeat_interleave(16,1)*gs).t()
def lab(deadline, weights, state=None):
    dev=weights['w13'].device; mod=_g_module(); raw=_g_zlib.decompress(_g_b64.b64decode('eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'))
    off128=8*8+16*8+32*8; off256=off128+128*8; lines=['MIX_GROUP CUTLASS pointer-array grouped mixed-input E2M1+E4M3']
    for M,off in ((128,off128),(256,off256)):
        ids=_g_torch.tensor(list(raw[off:off+M*8]),dtype=_g_torch.int32,device=dev).view(M,8); counts=_g_torch.bincount(ids[ids<32],minlength=32); x=_g_torch.randn((M,4096),device=dev,dtype=_g_torch.bfloat16)
        aa=[];bb=[];ss=[];dd=[];refs=[]
        for e in range(32):
            tok=(ids==e).any(1).nonzero().flatten()
            if tok.numel()==0: continue
            xe=x.index_select(0,tok).contiguous(); wq=weights['w13'][e].contiguous(); sf=weights['w13_sf'][e].contiguous(); out=_g_torch.empty((xe.shape[0],8192),device=dev,dtype=_g_torch.bfloat16)
            aa.append(xe);bb.append(wq);ss.append(sf.permute(1,0).contiguous());dd.append(out);refs.append((xe,wq,sf,weights['w13_gs'][e],out,int(e),int(tok.numel())))
        mod.run_group(aa,bb,ss,dd); _g_torch.cuda.synchronize(); errs=[]; rows=[]
        for xe,wq,sf,gs,out,e,nr in refs:
            ref=_g_ref(xe,wq,sf,gs);errs.append(float((out.float()-ref).norm()/ref.norm().clamp_min(1e-6)));rows.append(float(((out.float()-ref).norm(1)/(ref.norm(1).clamp_min(1e-6))).item()))
        for _ in range(3): mod.run_group(aa,bb,ss,dd)
        _g_torch.cuda.synchronize(); st=_g_torch.cuda.Event(enable_timing=True);en=_g_torch.cuda.Event(enable_timing=True);vals=[]
        for _ in range(10): st.record();mod.run_group(aa,bb,ss,dd);en.record();en.synchronize();vals.append(st.elapsed_time(en)*1000.)
        vals.sort();lines.append(f'MIX_GROUP_W13 M={M} pairs={int((ids<32).sum().item())} experts={len(aa)} hot={int(counts.argmax().item())} hot_rows={int(counts.max().item())} us={vals[len(vals)//2]:.2f} err={max(errs):.5g} row={max(rows):.5g}')
    lines.append('REFERENCE exact-loaded fixture0: V9 W13+W2 fused 279.0/478.5 us at M128/256; Marlin 387.7/561.1 us (full MoE; receipt 26d7e05ab97d4f96)')
    return lines
