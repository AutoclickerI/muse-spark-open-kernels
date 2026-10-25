"""Write prof/lab/vq8_layer.py: vq8 on the decode layer's ABI (FP32 projection partials in, [T, 1024] BF16 out),
combine kernel vs fused merge (flags bit 2), against the layer's incumbents on the same inputs: q_len 8 the verify
path (qk_prep + public FA3 + norm_gate), q_len 1 attn_main (sumcast + the attention port, from the decode lane's
S11 file). Every time is a chain: a projection stand-in, then the kernel (PDL), minus the stand-in alone; cold L2
by a read. Base: prof/lab/verify1_sliding_fa3_pair.py (the TASK gate, generate_input, the verify path)."""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
base = (ROOT / 'prof/lab/verify1_sliding_fa3_pair.py').read_text()
base = base[:base.index('def lab(deadline,weights):')]
S11 = ROOT / 'problems/muse_decode_layer/cand_r3_best7_s11.py'
s11 = S11.read_text()
PORT_NAMES = ('_attn_sumcast_kernel', '_attn_sumcast', 'ATTN_PORT_CUDA_SRC', 'ATTN_PORT_CPP_SRC', '_ATTN_PORT_EXT',
              '_ATTN_PORT_COUNTERS', '_attn_port_ext', '_attn_port_counters', '_attn_port_raw', '_attn_main_plan',
              '_attn_port_forward')


def _segments(src, names):
    lines = src.split('\n')
    found = {}
    for node in ast.parse(src).body:
        name = getattr(node, 'name', None) or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                                               and node.targets[0].id)
        if name in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            found[name] = '\n'.join(lines[start - 1:node.end_lineno])
    missing = set(names) - set(found)
    assert not missing, missing
    return '\n\n'.join(found[n] for n in names)


PORT = ('# ---- attn_main (q_len 1 incumbent) from ' + str(S11.relative_to(ROOT)) + ' SHA-256 '
        + hashlib.sha256(s11.encode()).hexdigest()[:16] + ' ----\nQKV_SIZE = 2304\n_LAB_SMS = {}\n\n\n'
        'def _num_sms(device):\n    idx = device.index if device.index is not None else torch.cuda.current_device()\n'
        '    if idx not in _LAB_SMS:\n        _LAB_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count\n'
        '    return _LAB_SMS[idx]\n\n\n' + _segments(s11, PORT_NAMES) + '\n\n\n'
        'def layer_attn_main(qkv, data):\n    return _attn_port_forward(_attn_port_raw(qkv, data), data)\n')
cu = (HERE / 'vq8.cu').read_text()
drv = (HERE / 'vq8.py').read_text()
CLOCK_SUMMARY = drv[drv.index('def clock_summary(rows, G):'):drv.index('def attention_vq8(')]

LAB = r'''
VQ8_CPP = r"""
#include <cstdint>
#include <vector>
void vq8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t);
std::vector<int64_t> vq8_resources(int64_t);
"""
_VQ8_MOD = {}
_VQ8_SMS = {}


def _vq8_build(clk):
    tag = hashlib.sha256(VQ8_CUDA.encode()).hexdigest()[:12]
    return load_inline(name=f'vq8lab{"clk" if clk else ""}_{tag}', cpp_sources=VQ8_CPP, cuda_sources=VQ8_CUDA,
                       functions=['vq8_forward', 'vq8_resources'],
                       extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-DVQ8_CLK={int(clk)}'],
                       verbose=False)


def _vq8_ext(clk=False):
    if clk not in _VQ8_MOD:
        _VQ8_MOD[clk] = _vq8_build(clk)
    return _VQ8_MOD[clk]


CLOCK_SUMMARY_SRC


def attention_vq8(qkv, data, setting=(8, 2), pdl=False, clocks=None, fused=False, part=None):
    """Decode ABI: qkv = (P, S), P the FP32 projection partials [S, T, W]. Returns BF16 [T, HQ*64].
    clocks: the clock build; its rows (after the partials) must be zero: pass a zeroed `part` (zeroed outside the
    timed chain) or let this call zero them (a memset between the stub and vq8)."""
    P, S = qkv
    cfg = data['config']
    B, L, HQ_, HK_ = cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads']
    dev = P.device
    if dev.index not in _VQ8_SMS:
        _VQ8_SMS[dev.index] = torch.cuda.get_device_properties(dev).multi_processor_count
    G = _VQ8_SMS[dev.index]
    NS = B * (HK_ // 2)
    nclk = (G + 16 * L * NS) * 64 if clocks is not None else 0
    if part is None:
        part = torch.empty(((G + NS) * 128 * 68 + 256 + nclk,), device=dev, dtype=torch.float32)
        if clocks is not None:
            part[(G + NS) * 128 * 68 + 256:].zero_()
    out = torch.empty((B * L, HQ_ * 64), device=dev, dtype=torch.bfloat16)
    window = cfg['window']
    invf = _inv_freq(cfg['rope_theta'], dev) if window else None
    _vq8_ext(clocks is not None).vq8_forward(
        P.data_ptr(), data['cache_loc'].data_ptr(), data['page_table'].data_ptr(), data['seq_lens'].data_ptr(),
        data['k_cache'].data_ptr(), data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), B, L, HQ_, HK_, P.stride(-2),
        data['page_table'].shape[1], cfg['page_size'], window, S, setting[1], G, setting[0],
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634, P.stride(0), int(pdl) | (4 if fused else 0),
        torch.cuda.current_stream(dev).cuda_stream)
    if clocks is not None:
        clocks.append(part[(G + NS) * 128 * 68 + 256:].view(torch.int64).view(G + 16 * L * NS, 32))
    return out


# Projection stand-in: streams a 37.7 MB weight (the QKV GEMV's bytes) and rewrites the partials P from a
# pristine copy (P = SRC + 0 * acc), triggering its dependents at entry as the real GEMV does.
@triton.jit
def _proj_stub(WT, SRC, DST, nw, ndst, zero, BLOCK: tl.constexpr, USE_PDL: tl.constexpr):
    pid = tl.program_id(0)
    npg = tl.num_programs(0)
    _pdl_start(USE_PDL)
    acc = tl.zeros((BLOCK,), tl.float32)
    for off in range(pid * BLOCK, nw, npg * BLOCK):
        acc += tl.load(WT + off + tl.arange(0, BLOCK), mask=off + tl.arange(0, BLOCK) < nw, other=0.).to(tl.float32)
    s = tl.sum(acc, 0) * zero
    _pdl_wait(USE_PDL)
    for off in range(pid * BLOCK, ndst, npg * BLOCK):
        idx = off + tl.arange(0, BLOCK)
        tl.store(DST + idx, tl.load(SRC + idx, mask=idx < ndst, other=0.) + s, mask=idx < ndst)


def _proj(wt, src, dst):
    _launch(_proj_stub, (132,), wt, src, dst, wt.numel(), dst.numel(), 0.0, BLOCK=4096, num_warps=8)


def timed_cold(fn, data, rounds=7):
    fn(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(rounds):
        lab_cold_l2()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        vals.append(start.elapsed_time(end) * 1000.)
    return sorted(vals)[len(vals) // 2], out, graph


def lab(deadline, weights):
    import time as _time
    lines = ['META vq8 (decode ABI, 4 FP32 partials) combine kernel (c) vs fused merge (f) vs the layer incumbent: '
             'q_len 8 qk_prep + public FA3 + norm_gate, q_len 1 attn_main (sumcast + port); ctx 80000 page 1, cold L2 '
             'by read; us = chain (37.7 MB projection stand-in, then the kernel with PDL) minus the stand-in alone',
             'META ROW shape kernel chain_us gate(TASK check incl. new K/V) err; vq8 settings (stages, min_tiles)']
    t0 = _time.monotonic()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(3) as pool:  # the production and phase-clock builds and the port compile together
        port = pool.submit(_attn_port_ext)
        for clk, mod in zip((False, True), pool.map(_vq8_build, (False, True))):
            _VQ8_MOD[clk] = mod
        port.result()
    lines.append('COMPILE_S %.1f res st6=%s' % (_time.monotonic() - t0, _vq8_ext().vq8_resources(6)))
    S = 4
    wt = torch.randn((37748736 // 2,), device='cuda').to(torch.bfloat16)
    order = ((1, 'sliding', 8), (1, 'full', 8), (1, 'sliding', 1), (1, 'full', 1), (16, 'sliding', 8),
             (32, 'sliding', 8), (16, 'sliding', 1), (32, 'sliding', 1), (16, 'full', 8), (32, 'full', 8),
             (16, 'full', 1), (32, 'full', 1))
    for B, layer, L in order:
        if deadline - _time.monotonic() < 6:
            lines.append('SKIP B=%d %s L=%d budget' % (B, layer, L))
            break
        try:
            T = B * L
            data = generate_input(B, 80000, 'tp8', layer, 4100 + B, q_len=L, page=1,
                                  dist='soft' if layer == 'sliding' else 'coherent', spread='agentic' if B > 1 else 'fixed')
            data['config']['tokens'] = T
            theta = data['config']['rope_theta']
            state = {'inv_freq': 1. / (theta ** (torch.arange(0, 64, 2, device='cuda').float() / 64.)) if theta else
                     torch.ones(32, device='cuda')}
            g = torch.Generator(device='cuda').manual_seed(B)
            parts = torch.randn((S, T, 2304), generator=g, device='cuda') * 0.5
            parts[S - 1] = data['qkv'].view(T, 2304).float() - parts[:S - 1].sum(0)  # the partials sum to the input
            data['qkv'] = parts.sum(0).to(torch.bfloat16).view(B, L, 2304)
            src = parts.clone()
            qkv = (parts, S)
            layer_fa3(qkv, data, state, glue='current')  # the step's new K/V in the cache, then the TASK reference
            torch.cuda.synchronize()
            expected = gate_prepare(data)
            if L == 8:
                kernels = [('fa3', lambda d: layer_fa3(qkv, d, state, glue='current'), None)]
            else:
                kernels = [('attn_main', lambda d: layer_attn_main(qkv, d), None)]
            for st in ((4, 1), (4, 2), (6, 8)):
                for fz in (False, True):
                    kernels.append(('vq8%s_st%dmt%d' % ('f' if fz else 'c', *st),
                                    lambda d, st=st, fz=fz: attention_vq8(qkv, d, st, pdl=True, fused=fz), (st, fz)))
            stub_us, _, _ = timed_cold(lambda d: _proj(wt, src, parts), data)
            best = {}
            row = []
            for name, fn, key in kernels:
                if deadline - _time.monotonic() < 3:
                    row.append('SKIP')
                    break
                us, out, _ = timed_cold(lambda d, fn=fn: (_proj(wt, src, parts), fn(d))[1], data)
                ok, msg, err = gate_cached(data, out, expected)
                row.append('%s=%.2f%s' % (name, us - stub_us, '' if ok else '(F %.3f)' % err))
                if not ok:
                    lines.append('GATE_ERROR %s %s' % (name, msg[:120]))
                if key is not None and ok and (key[1] not in best or us < best[key[1]][1]):
                    best[key[1]] = (key[0], us)
            lines.append('ROW B=%d %s L=%d stub=%.1f %s' % (B, layer, L, stub_us, ' '.join(row)))
            for fz in (False, True):  # phase clocks of the best setting of each merge, in the chain
                if fz in best and deadline - _time.monotonic() > 2:
                    G, NS = _VQ8_SMS[0], B * (data['config']['num_kv_heads'] // 2)
                    part = torch.zeros(((G + NS) * 128 * 68 + 256 + (G + 16 * L * NS) * 64,), device='cuda')
                    rows = []
                    lab_cold_l2()
                    _proj(wt, src, parts)  # the stub, then vq8 with PDL (no clock zeroing between them)
                    attention_vq8(qkv, data, best[fz][0], pdl=True, clocks=rows, fused=fz, part=part)
                    torch.cuda.synchronize()
                    lines.append('CLK B=%d %s L=%d %s st%dmt%d %s' % (B, layer, L, 'f' if fz else 'c', *best[fz][0],
                                                                    clock_summary(rows[0], _VQ8_SMS[0])))
        except Exception as exc:  # a failed shape keeps the others
            lines.append('SHAPE_ERROR B=%d %s L=%d %s' % (B, layer, L, str(exc).splitlines()[0][:160]))
        data = parts = src = expected = qkv = None
        torch.cuda.empty_cache()
    lines.append('LAB_S %.1f' % (_time.monotonic() - t0))
    return lines
'''

LAB = LAB.replace('CLOCK_SUMMARY_SRC', CLOCK_SUMMARY.rstrip())
out = ('"""vq8 (combine or fused merge) vs the decode layer attention incumbents on its ABI (generated by problems/muse_attention_decode/'
       'dev/vq8/build_layer_lab.py; vq8.cu SHA-256 ' + hashlib.sha256(cu.encode()).hexdigest() + ')."""\n'
       + base + '\n' + PORT + '\nimport hashlib\nVQ8_CUDA = ' + repr(cu) + '\n' + LAB)
(ROOT / 'prof/lab/vq8_layer.py').write_text(out)
print('wrote prof/lab/vq8_layer.py', len(out))
