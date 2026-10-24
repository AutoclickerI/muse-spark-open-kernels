"""CUTLASS SM90 mixed-input FP4 probe; weights remain packed and scales stay E4M3."""
import base64 as _mix_b64
import zlib as _mix_zlib
import torch as _mix_torch
from torch.utils.cpp_extension import load_inline as _mix_load_inline

_MIX_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
using namespace cute;
using EA = cutlass::bfloat16_t;
using EB = cute::tuple<cutlass::float_e2m1_t, cutlass::float_e4m3_t>;
using TS = cute::Shape<cute::_128,cute::_128,cute::_64>;
using CS = cute::Shape<cute::_1,cute::_1,cute::_1>;
using ML = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    EA, cutlass::layout::RowMajor, 8,
    EB, cutlass::layout::ColumnMajor, 32,
    float, TS, CS, cutlass::gemm::collective::StageCountAuto, cutlass::gemm::KernelTmaWarpSpecialized>::CollectiveOp;
using EL = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TS, CS, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::bfloat16_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    cutlass::epilogue::fusion::LinearCombination<cutlass::bfloat16_t,float>>::CollectiveOp;
using GK = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,ML,EL,void>;
using GA = cutlass::gemm::device::GemmUniversalAdapter<GK>;
void run_gemm(torch::Tensor a, torch::Tensor b, torch::Tensor s, torch::Tensor d) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && s.is_cuda() && d.is_cuda());
  int M=(int)a.size(0), K=(int)a.size(1), N=(int)d.size(1);
  using S3 = cute::Stride<int64_t,cute::Int<1>,int64_t>;
  using SB = cute::Stride<int64_t,cute::Int<1>,int64_t>;
  using SS = cute::Stride<cute::Int<1>,int64_t,int64_t>;
  S3 da{K,cute::Int<1>{},0};
  SB db{K,cute::Int<1>{},0};
  SS ds{cute::Int<1>{},N,0};
  ML::Arguments ma{};
  ma.ptr_A = reinterpret_cast<const EA*>(a.data_ptr()); ma.dA=da;
  ma.ptr_B = reinterpret_cast<const cutlass::float_e2m1_t*>(b.data_ptr()); ma.dB=db;
  ma.ptr_S = reinterpret_cast<const cutlass::float_e4m3_t*>(s.data_ptr()); ma.dS=ds; ma.group_size=16;
  typename EL::Arguments ea{};
  ea.thread = typename EL::ThreadEpilogueOp::Params(1.f,0.f);
  ea.ptr_C = nullptr; ea.dC = {int64_t(0), cute::Int<1>{}, int64_t(0)};
  ea.ptr_D = reinterpret_cast<cutlass::bfloat16_t*>(d.data_ptr()); ea.dD = {int64_t(N), cute::Int<1>{}, int64_t(0)};
  typename GA::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, Shape<int,int,int,int>{M,N,K,1}, ma, ea};
  auto st=GA::can_implement(args); TORCH_CHECK(st==cutlass::Status::kSuccess,"cannot implement ",int(st));
  GA op; auto stream=at::cuda::getCurrentCUDAStream();
  st=op.initialize(args,nullptr,stream); TORCH_CHECK(st==cutlass::Status::kSuccess,"init ",int(st));
  st=op.run(stream); TORCH_CHECK(st==cutlass::Status::kSuccess,"run ",int(st));
}
"""
_MIX_MOD = None

def _mix_module():
    global _MIX_MOD
    if _MIX_MOD is None:
        _MIX_MOD = _mix_load_inline(
            name='muse_cutlass_mixed_input_probe_84b94286',
            cpp_sources='void run_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);',
            cuda_sources=_MIX_CUDA,
            functions=['run_gemm'],
            extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/deep_gemm/include'],
            extra_cflags=['-std=c++20','-O3'],
            extra_cuda_cflags=['-std=c++20','-O3','-gencode=arch=compute_90a,code=sm_90a'],
            verbose=False)
    return _MIX_MOD

def _mix_ref(x, wq, sf, gs):
    grid = _mix_torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=x.device)
    c = _mix_torch.stack((wq & 15, wq >> 4), -1).reshape(wq.shape[0], -1)
    mag = grid[(c & 7).long()]
    v = _mix_torch.where((c & 8) != 0, -mag, mag)
    ww = (v * sf.float().repeat_interleave(16, dim=1) * gs).t().contiguous()
    return x.float() @ ww

def lab(deadline, weights, state=None):
    dev = weights['w13'].device
    mod = _mix_module()
    raw = _mix_zlib.decompress(_mix_b64.b64decode('eNp1V4tu5CAM3F01PVSp6DjvSk1s7f//pS9gjG1IK7VJwBg/ZgbK/MPMmfvPH3m070w/9nG+Pup7G3qr+TkGzMTyy4ztF2SuPYiXjx3lVYa6cf1GdZFsHau3OgzDvloBBrvmBMOmOJaj7If1L9YNyC9qf33UpwG0NXVLTsSlx3omR+K3GSaUdejD7AUQr2ROUUODKS8L9ljq0h2muYx05SK5uraFrVE7h1q2uCXfngeOZrbXc9FHW1+zan6eMv+Zxja2dXspvZRvftfJe+gcTkmavfMDlgM0g9DIkUXqLjLF0kHo+JhEKcO59oFw31wUtb45pj9WHVZAJlyqt5/lGQPE0+r63OrUxp813sS//FT0Cn8cNL55bantkW0j1xeL8DkK9Tr9lsCM/qC6RcLRcK08zZXkCwJmhJlcisNjxHFu/OhWjbB1tqxoJcmHwvhu3WujN0EL+MbgTh0kjgQphEqeQwP+gNb/Wkn6hgmdfJF754jxsVXQIbMZHGvZpJ0jGSMMDUyUFxufB5HAe/Y4/5akXmHnWq+9ryIfs/XDrDsk4CvkSE3iHBAeMrH1GpZJj4bDzY2Tej88JmGBEk1FggHzwgIVv+blnGh8mwqlnRwpx62aLA+Le6hj1W+ZTBdlErv3POCfjVkpMCuCaHMKf5qXGoAw5626+lQZpMg/mRHV7VNvw0G66IHAHz0Aeg8c9D8nJX5Wy/sIsiSj0pR2r+PhyguKjUnUAwqY/0kyl/rnigoTIgIHUE0ftYykKWmqx6QgYNeSjp6/NIKFjiZHrPSt/Qh50Gbs90BOEx9b6HpIwq7r85V+epiQowJdyS12DOeW8M0VDZeLVf1spyGkpcQfTvn9gZ00ElTJQEWXmZWIgzVQ0QeOyPM6ETgHJjae+F5XHQCoi52LB/0OOQSETvuC3i10+ZLxzKHx/rA1Ag2oPTyNFP39YhB1McWu3yie6PUiQvzL7W8rJ9Xh6pj09qSCIDzJpHAR2fe03PPtl3Pc2uDG8wHWzw+PXHAluIoPwllJ09kZJXbIfa2L3NbVYT7WG3G5uiUaKoCEV15Dyw0uEEvjAD0XoXaFxv1/NBPkn4NOz9c4EWkFFImME07/g2ioKgQFDqdatBxoXcYVwZR6iYAmDiVwmBV3/wFW3hUK'))
    # Fixture order is M=8,16,32,128,256; route IDs are global and local_start=0.
    off128 = 8*8 + 16*8 + 32*8
    off256 = off128 + 128*8
    lines = ['MIX_CUTLASS mixed-input E2M1->BF16 with E4M3 group16 scale']
    for M, off in ((128, off128), (256, off256)):
        ids = _mix_torch.tensor(list(raw[off:off+M*8]), dtype=_mix_torch.int32, device=dev).view(M,8)
        counts = _mix_torch.bincount(ids[ids < 32], minlength=32)
        hot = int(counts.argmax().item())
        tok = (ids == hot).any(dim=1).nonzero().flatten()
        x = _mix_torch.randn((M,4096), device=dev, dtype=_mix_torch.bfloat16)
        xe = x.index_select(0, tok).contiguous()
        wq = weights['w13'][hot].contiguous()
        sf = weights['w13_sf'][hot].contiguous()
        gs = weights['w13_gs'][hot]
        # One-time gate against independent FP32 dequantized reference; no derived bank survives.
        d = _mix_torch.empty((xe.shape[0],8192), device=dev, dtype=_mix_torch.bfloat16)
        mod.run_gemm(xe, wq, sf.permute(1,0).contiguous(), d)
        _mix_torch.cuda.synchronize()
        ref = _mix_ref(xe, wq, sf, gs)
        err = (d.float()-ref).norm().div(ref.norm().clamp_min(1e-6)).item()
        row = ((d.float()-ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-6)).max().item()
        for _ in range(3): mod.run_gemm(xe,wq,sf.permute(1,0).contiguous(),d)
        _mix_torch.cuda.synchronize()
        st = _mix_torch.cuda.Event(enable_timing=True); en = _mix_torch.cuda.Event(enable_timing=True)
        vals=[]
        for _ in range(10):
            st.record(); mod.run_gemm(xe,wq,sf.permute(1,0).contiguous(),d); en.record(); en.synchronize(); vals.append(st.elapsed_time(en)*1000.)
        vals.sort(); med=vals[len(vals)//2]
        lines.append(f'MIX_W13 M={M} hot={hot} rows={xe.shape[0]} experts={int((counts>0).sum().item())} us={med:.2f} err={err:.5g} row={row:.5g}')
    return lines
