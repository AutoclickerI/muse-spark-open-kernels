"""Driver for vq8 (q_len 8 attention, both KV heads per CTA, stream-K, native FP8 WGMMA on sm_90a)."""
import hashlib
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

VQ8_CUDA = (Path(__file__).resolve().parent / 'vq8.cu').read_text()
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
        _EXT[key] = load_inline(name=f'vq8{"clk" if clk else ""}_{tag}_{arch}', cpp_sources=VQ8_CPP,
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
    return cfg['q_len'] in (1, 8) and cfg['head_dim'] == 64 and hq == 8 * hk and hk % 2 == 0 and cfg['batch'] * hk // 2 <= 128


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
    G = grid or _num_sms(device)
    NS = B * (HK // 2)
    L = cfg['q_len']
    nclk = (G + 16 * L * NS) * 64 if clocks is not None else 0  # up to 8 warps per combine row
    part = torch.empty(((G + NS) * 128 * 68 + 256 + nclk,), device=device, dtype=torch.float32)
    out = torch.empty((B, L, HQ * 64), device=device, dtype=torch.bfloat16)
    window = cfg['window']
    invf = _inv_freq(cfg['rope_theta'], device) if window else None
    if clocks is not None:
        part[(G + NS) * 128 * 68 + 256:].zero_()
    vq8_ext(clocks is not None).vq8_forward(
        raw.data_ptr(), data['cache_loc'].data_ptr(), data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), B, L, HQ, HK, raw.stride(-2),
        data['page_table'].shape[1], cfg['page_size'], window, proj_splits, min_tiles, G, stages,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634, split_stride, int(pdl) | (0 if gate else 2) | (4 if fused else 0),
        torch.cuda.current_stream(device).cuda_stream)
    if clocks is not None:
        clocks.append(part[(G + NS) * 128 * 68 + 256:].view(torch.int64).view(G + 16 * L * NS, 32))
    return out


def clock_summary(rows, G):
    """One line from vq8_attention(clocks=...) rows: times in us from the first CTA's entry, wait shares."""
    rows = rows.cpu()
    main, comb = rows[:G], rows[G:]
    comb = comb[comb[:, 0] != 0]
    fused = ''
    if len(comb) == 0:  # fused merge: rows 25 barrier release, 27 slice loaded, 28 slices merged, 26 end (thread 0)
        mrg = main[main[:, 26] != 0]
        comb = torch.stack([mrg[:, 25], mrg[:, 25], mrg[:, 26]], 1)
        live = mrg[mrg[:, 27] != 0]
        d = lambda c: float(((live[:, c] & ((1 << 62) - 1)) - live[:, 25]).double().median()) / 1000.0  # noqa: E731
        fused = ' merge_us slice=%.2f red=%.2f end=%.2f' % (d(27), d(28), d(26)) if len(live) else ''
    act = main[main[:, 10] != -1]
    t0 = int(main[:, 0].min())
    us = lambda x: (x.double() - t0) / 1000.0  # noqa: E731
    med = lambda x: float(x.double().median())  # noqa: E731
    cyc_c, cyc_p = act[:, 5].double().clamp(min=1), act[:, 9].double().clamp(min=1)
    return ("ctas=%d tiles=%d-%d entry_max=%.2f sched=%.2f first=%.2f/%.2f wg0_end=%.2f/%.2f wg1_end_max=%.2f "
            "prod_end_max=%.2f cwait=%.2f/%.2f pempty=%.2f pcp=%.2f comb_entry=%.2f comb_go=%.2f comb_end=%.2f "
            "cyc_per_tile=%.0f cons_go=%.2f bar1=%.2f bar2=%.2f loads=%.2f q_ready=%.2f "
            "per-tile cyc qkw=%.0f smx=%.0f pvw=%.0f iss=%.0f pub=%.0f pis=%.0f" % (
                len(act), int(act[:, 10].min()), int(act[:, 10].max()), float(us(main[:, 0]).max()),
                med(us(act[:, 1]) - us(act[:, 0])), med(us(act[:, 2])), float(us(act[:, 2]).max()),
                med(us(act[:, 3])), float(us(act[:, 3]).max()), float(us(act[:, 13]).max()), float(us(act[:, 6]).max()),
                med(act[:, 4] / cyc_c), med(act[:, 11] / cyc_c), med(act[:, 7] / cyc_p), med(act[:, 8] / cyc_p),
                float(us(comb[:, 0]).min()), float(us(comb[:, 1]).min()), float(us(comb[:, 2]).max()),
                med(cyc_c / act[:, 10].double().clamp(min=1)), med(us(act[:, 14])), med(us(act[:, 16])), med(us(act[:, 17])),
                med(us(act[:, 18])), med(us(act[:, 15])),
                *[med(act[:, c].double() / act[:, 10].double().clamp(min=1)) for c in range(19, 25)])) + fused


def attention_vq8(qkv, data, setting=(8, 2), pdl=False, gate=True, fused=False):
    """Decode-layer ABI: qkv = (P, S), P the QKV projection's partials [S, T, W] (FP32) or BF16 with S == 1.
    Returns bfloat16 [T, HQ*64] (the O projection's input); writes the step's K/V into the cache."""
    P, S = qkv
    cfg = data['config']
    if P.dtype == torch.bfloat16:
        assert S == 1
        raw = P[0] if P.dim() == 3 else P
        out = vq8_attention(data, qkv=raw, stages=setting[0], min_tiles=setting[1], pdl=pdl, gate=gate, fused=fused)
    else:
        assert P.dtype == torch.float32 and P.stride(-1) == 1
        out = vq8_attention(data, qkv=P, proj_splits=S, split_stride=P.stride(0), stages=setting[0],
                            min_tiles=setting[1], pdl=pdl, gate=gate, fused=fused)
    return out.view(cfg['batch'] * cfg['q_len'], cfg['num_q_heads'] * 64)
