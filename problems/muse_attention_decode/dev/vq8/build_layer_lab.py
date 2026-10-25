"""Write prof/lab/vq8_layer.py: vq8 on the decode layer's ABI (FP32 projection partials in, [T, 1024] BF16 out),
combine kernel vs fused merge (flags bit 2), against the layer's incumbents on the same inputs: q_len 8 the verify
path (qk_prep + public FA3 + norm_gate), q_len 1 attn_main (sumcast + the attention port, from the decode lane's
S11 file). Every time is a chain: a projection stand-in, then the kernel (PDL), minus the stand-in alone; cold L2
by a read. Base: prof/lab/verify1_sliding_fa3_pair.py (the TASK gate, generate_input, the verify path)."""
import ast
import argparse
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
base = (ROOT / 'prof/lab/verify1_sliding_fa3_pair.py').read_text()
base = base[:base.index('def lab(deadline,weights):')]
S11 = ROOT / 'problems/muse_decode_layer/cand_r3_best7_s11_final.py'
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
ap = argparse.ArgumentParser()
ap.add_argument('--fastprep', action='store_true')
ap.add_argument('--softmax', action='store_true')
ap.add_argument('--compact', action='store_true')
args = ap.parse_args()
if args.softmax:
    from build_softmax import source
    cu = source()
elif args.fastprep:
    from build_fastprep import source
    cu = source()
else:
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
_VQ8_OLD = None
_VQ8_SMS = {}


def _vq8_build(clk, old=False):
    source = VQ8_OLD_CUDA if old else VQ8_CUDA
    tag = hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name=f'vq8lab{"clk" if clk else ""}_{tag}', cpp_sources=VQ8_CPP, cuda_sources=source,
                       functions=['vq8_forward', 'vq8_resources'],
                       extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', f'-DVQ8_CLK={int(clk)}'],
                       verbose=False)


def _vq8_ext(clk=False):
    if clk not in _VQ8_MOD:
        _VQ8_MOD[clk] = _vq8_build(clk)
    return _VQ8_MOD[clk]


CLOCK_SUMMARY_SRC


def attention_vq8(qkv, data, setting=(8, 2), pdl=False, clocks=None, fused=False, part=None, old=False):
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
    (_VQ8_OLD if old else _vq8_ext(clocks is not None)).vq8_forward(
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


def paired(kernels, data, rounds=8):
    graphs, outputs = [], []
    for name, fn, _ in kernels:
        fn(data)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn(data)
        graphs.append(graph)
        outputs.append(out)
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    times = [[] for _ in kernels]
    ratios = [[] for _ in kernels]
    def run(i):
        lab_cold_l2()
        ev[0].record(); graphs[i].replay(); ev[1].record(); ev[1].synchronize()
        return ev[0].elapsed_time(ev[1]) * 1000.
    for i in range(1, len(kernels)):
        for r in range(rounds):
            order = (0, i) if r % 2 == 0 else (i, 0)
            vals = {j: run(j) for j in order}
            for j in order: times[j].append(vals[j])
            ratios[i].append(vals[i] / vals[0])
    return times, ratios, outputs


def lab(deadline, weights):
    import time as _time
    import statistics as _st
    global _VQ8_OLD
    lines = ['META paired AB/BA x8, 3 seeds; identical projection-partials and KV per arm, read-flushed L2; '
             'ratio=variant/control entire projection-stub+attention graph. Control: final S11 attn_main at L1, '
             'public FA3 at L8 sliding/B1 full, frozen c262e4c5 vq8 at L8 B16/32 full.',
             'META us are chain minus the same projection stub; TASK gate including cache; local FA3 stub fails by design.']
    t0 = _time.monotonic()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as pool:
        port = pool.submit(_attn_port_ext)
        old = pool.submit(_vq8_build, False, True)
        for clk, mod in zip((False, True), pool.map(_vq8_build, (False, True))):
            _VQ8_MOD[clk] = mod
        _VQ8_OLD = old.result()
        port.result()
    lines.append('COMPILE_S %.1f res=%s' % (_time.monotonic() - t0, _vq8_ext().vq8_resources(4)))
    summaries, clocks = {}, []
    wt = torch.randn((37748736 // 2,), device='cuda').to(torch.bfloat16)
    order = ((1, 'sliding', 8), (1, 'sliding', 1), (16, 'sliding', 8), (32, 'sliding', 8),
             (16, 'sliding', 1), (32, 'sliding', 1), (16, 'full', 1), (32, 'full', 1),
             (16, 'full', 8), (32, 'full', 8), (1, 'full', 1), (1, 'full', 8))
    for seed in (4100, 5100, 6100):
        for B, layer, L in order:
            if deadline - _time.monotonic() < 6:
                lines.append('SKIP seed=%d B=%d %s L=%d budget' % (seed, B, layer, L))
                break
            try:
                T, S = B * L, 4
                data = generate_input(B, 80000, 'tp8', layer, seed + B, q_len=L, page=1,
                                      dist='soft' if layer == 'sliding' else 'coherent', spread='agentic' if B > 1 else 'fixed')
                data['config']['tokens'] = T
                theta = data['config']['rope_theta']
                state = {'inv_freq': 1. / (theta ** (torch.arange(0, 64, 2, device='cuda').float() / 64.)) if theta else
                         torch.ones(32, device='cuda')}
                g = torch.Generator(device='cuda').manual_seed(seed + B)
                parts = torch.randn((S, T, 2304), generator=g, device='cuda') * 0.5
                parts[-1] = data['qkv'].view(T, 2304).float() - parts[:-1].sum(0)
                data['qkv'] = parts.sum(0).to(torch.bfloat16).view(B, L, 2304)
                src, qkv = parts.clone(), (parts, S)
                # Populate the new slots before the untimed TASK reference.
                attention_vq8(qkv, data, (4, 1))
                torch.cuda.synchronize()
                expected = gate_prepare(data)
                if L == 1:
                    control = lambda d: layer_attn_main(qkv, d)
                    control_name = 'attn_main'
                elif B > 1 and layer == 'full':
                    control = lambda d: attention_vq8(qkv, d, (6, 8), pdl=True, old=True)
                    control_name = 's11_vq8'
                else:
                    control = lambda d: layer_fa3(qkv, d, state, glue='current')
                    control_name = 'fa3'
                kernels = [(control_name, control, None)]
                for st in ((4, 1), (6, 8)):
                    for fz in (False, True):
                        kernels.append(('%s%d' % ('f' if fz else 'c', st[0]),
                                        lambda d, st=st, fz=fz: attention_vq8(qkv, d, st, pdl=True, fused=fz), (st, fz)))
                stub_us, _, _ = timed_cold(lambda d: _proj(wt, src, parts), data)
                chained = [(name, lambda d, fn=fn: (_proj(wt, src, parts), fn(d))[1], key) for name, fn, key in kernels]
                times, ratios, outs = paired(chained, data)
                for i, (name, fn, key) in enumerate(kernels):
                    fn(data); torch.cuda.synchronize()  # cache check belongs to this arm
                    ok, msg, err = gate_cached(data, outs[i], expected)
                    skey = (B, layer, L, name)
                    summaries.setdefault(skey, []).append((_st.median(times[i]) - stub_us,
                                                           _st.median(ratios[i]) if i else 1., ok, err))
                    if not ok: lines.append('GATE_ERROR %s %s' % (skey, msg[:100]))
                if seed == 4100 and (layer == 'full' and L == 8 or B == 1 and layer == 'sliding'):
                    for fz in (False, True):
                        st = (6, 8) if layer == 'full' else (4, 1)
                        G, NS = _VQ8_SMS[0], B * (data['config']['num_kv_heads'] // 2)
                        part = torch.zeros(((G + NS) * 128 * 68 + 256 + (G + 16 * L * NS) * 64,), device='cuda')
                        rows = []
                        lab_cold_l2(); _proj(wt, src, parts)
                        attention_vq8(qkv, data, st, pdl=True, clocks=rows, fused=fz, part=part)
                        torch.cuda.synchronize()
                        clocks.append('CLK %d%s%d %s %s' % (B, layer[0], L, 'f' if fz else 'c', clock_summary(rows[0], G)))
            except Exception as exc:
                lines.append('SHAPE_ERROR seed=%d B=%d %s L=%d %s' % (seed, B, layer, L, str(exc).splitlines()[0][:160]))
            data = parts = src = expected = qkv = outs = kernels = chained = None
            torch.cuda.empty_cache()
    lines += clocks
    for (B, layer, L, name), rows in summaries.items():
        lines.append('PAIR %d%s%d %s n=%d us=%.2f ratio=%.4f[%.4f,%.4f] gate=%s err=%.4f' % (
            B, layer[0], L, name, len(rows), _st.median(x[0] for x in rows), _st.median(x[1] for x in rows),
            min(x[1] for x in rows), max(x[1] for x in rows), 'P' if all(x[2] for x in rows) else 'F', max(x[3] for x in rows)))
    lines.append('LAB_S %.1f' % (_time.monotonic() - t0))
    return lines

'''

if args.softmax:
    LAB = LAB.replace('for st in ((4, 1), (6, 8)):', 'for st in ((4, 1), (6, 8), (8, 8)):').replace('for fz in (False, True):', 'for fz in (False,):')
if args.compact:
    LAB = LAB.replace("    NS = B * (HK_ // 2)\n", "    NS = B * (HK_ // 2)\n    if cfg['window']:\n        G = min(G,max(1,NS*((cfg['window']+63)//64+1)//setting[1]))\n")
    LAB = LAB.replace("part[(G + NS) * 128 * 68 + 256:].view(torch.int64)", "part[(G + NS) * 128 * 68 + 256:(G + NS) * 128 * 68 + 256 + nclk].view(torch.int64)")
    LAB = LAB.replace("clock_summary(rows[0], G)", "clock_summary(rows[0],rows[0].shape[0]-16*L*NS)")
if args.softmax or args.compact:
    lo=LAB.index('    order = (('); hi=LAB.index('    for seed in (4100, 5100, 6100):',lo)
    order=(((16,'full',8),(32,'full',8),(16,'full',1),(32,'full',1),(1,'sliding',8),(1,'sliding',1)) if args.softmax else
           tuple((b,'sliding',L) for L in (1,8) for b in (1,16,32))+((16,'full',8),(32,'full',8)))
    LAB=LAB[:lo]+'    order = '+repr(order)+'\n'+LAB[hi:]
    if args.softmax:
        LAB=LAB.replace("if seed == 4100 and (layer == 'full' and L == 8 or B == 1 and layer == 'sliding'):",
                        "if seed == 4100 and layer == 'full' and L == 8:")
    else:
        LAB=LAB.replace('for st in ((4, 1), (6, 8)):', "for st in (((4,1),) if layer=='sliding' else ((6,8),)):")
        LAB=LAB.replace("if seed == 4100 and (layer == 'full' and L == 8 or B == 1 and layer == 'sliding'):",
                        "if seed == 4100 and (B == 16 and layer == 'full' and L == 8 or B == 1 and layer == 'sliding'):")
        LAB=LAB.replace('for fz in (False, True):\n                        st =',
                        "for fz in (False, True):\n                        if (layer == 'sliding' and not fz) or (layer == 'full' and fz):continue\n                        st =")
LAB = LAB.replace('CLOCK_SUMMARY_SRC', CLOCK_SUMMARY.rstrip())
out = ('"""vq8 (combine or fused merge) vs the decode layer attention incumbents on its ABI (generated by problems/muse_attention_decode/'
       'dev/vq8/build_layer_lab.py; vq8.cu SHA-256 ' + hashlib.sha256(cu.encode()).hexdigest() + ')."""\n'
       + base + '\nVQ8_OLD_CUDA = ' + repr((HERE / 'out/vq8_c262e4c5.cu').read_text()) + '\n' + PORT + '\nimport hashlib\nVQ8_CUDA = ' + repr(cu) + '\n' + LAB)
target = ROOT / ('prof/lab/vq8_softmax.py' if args.softmax else 'prof/lab/vq8_fastprep2.py' if args.compact else 'prof/lab/vq8_fastprep.py' if args.fastprep else 'prof/lab/vq8_layer.py')
target.write_text(out.rstrip() + '\n')
print('wrote', target, len(out))
