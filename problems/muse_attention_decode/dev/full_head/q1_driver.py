"""Driver for vq8 (q_len 8 attention, one KV head per CTA, stream-K, native FP8 WGMMA on sm_90a)."""
import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

import build_q1
VQ8_CUDA=build_q1.source()
VQ8_CPP = r'''
#include <cstdint>
#include <vector>
void vq8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t);
std::vector<int64_t> vq8_resources(int64_t);
'''
_EXT = {}
_SMS = {}
_INVF = {}


def vq8_ext(clk=False):
    """The extension; clk=True builds the phase-clock variant (VQ8_CLK, lab only)."""
    major, minor = torch.cuda.get_device_capability()
    arch = '90a' if major == 9 else f'{major}{minor}'
    key = (arch, clk)
    if key not in _EXT:
        tag = hashlib.sha256(VQ8_CUDA.encode()).hexdigest()[:12]
        _EXT[key] = load_inline(name=f'fullheadq1{"clk" if clk else ""}_{tag}_{arch}', cpp_sources=VQ8_CPP,
                                cuda_sources=VQ8_CUDA, functions=['vq8_forward', 'vq8_resources'],
                                extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}', '-lineinfo',
                                                   f'-DVQ8_CLK={int(clk)}'],
                                verbose=False)
    return _EXT[key]


def _num_sms(device):
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SMS:
        _SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SMS[idx]


def _inv_freq(theta, device):
    key = (float(theta), str(device))
    if key not in _INVF:
        _INVF[key] = (1.0 / (theta ** (torch.arange(0, 64, 2, device=device).float() / 64))).contiguous()
    return _INVF[key]


def vq8_supported(cfg):
    hq, hk = cfg['num_q_heads'], cfg['num_kv_heads']
    return cfg['q_len'] in (1, 8) and cfg['head_dim'] == 64 and hq == 8 * hk and cfg['batch'] * hk <= 64


def vq8_attention(data, qkv=None, proj_splits=0, split_stride=0, min_tiles=2, stages=8, grid=None, pdl=False,
                  clocks=None, gate=True, fused=False):
    """Returns bfloat16 [B, L, HQ*64] and writes the block's new K/V into the cache. qkv: BF16 [.., W]
    (default data['qkv']) or, with proj_splits > 0, FP32 projection partials (split i at i*split_stride).
    clocks: a list that receives the VQ8_CLK build's int64 rows [G + 16L*NS, 32] (see vq8.cu).
    gate=False: RMSNorm only (the gate split applies sigmoid(gate) in its own kernel; qkv's gate columns unread).
    fused=True: the main grid merges its partials after a grid barrier (no combine kernel)."""
    cfg = data['config']
    B, HQ, HK = cfg['batch'], cfg['num_q_heads'], cfg['num_kv_heads']
    raw = data['qkv'] if qkv is None else qkv
    device = raw.device
    if torch.cuda.get_device_capability(device)[0] != 9:
        stages = 2  # sm_120 screen: 99 KB of shared memory per block
    G = grid or 2*_num_sms(device)
    NS = B * HK
    L = cfg['q_len']
    nclk = (G + 16 * L * NS) * 64 if clocks is not None else 0  # up to 8 warps per combine row
    part = torch.empty(((G + NS) * 8 * 68 + 256 + nclk,), device=device, dtype=torch.float32)
    out = torch.empty((B, L, HQ * 64), device=device, dtype=torch.bfloat16)
    window = cfg['window']
    invf = _inv_freq(cfg['rope_theta'], device) if window else None
    if clocks is not None:
        part[(G + NS) * 8 * 68 + 256:].zero_()
    vq8_ext(clocks is not None).vq8_forward(
        raw.data_ptr(), data['cache_loc'].data_ptr(), data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), B, L, HQ, HK, raw.stride(-2),
        data['page_table'].shape[1], cfg['page_size'], window, proj_splits, min_tiles, G, stages,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634, split_stride, int(pdl) | (0 if gate else 2) | (4 if fused else 0),
        torch.cuda.current_stream(device).cuda_stream)
    if clocks is not None:
        clocks.append(part[(G + NS) * 8 * 68 + 256:].view(torch.int64).view(G + 16 * L * NS, 32))
    return out


