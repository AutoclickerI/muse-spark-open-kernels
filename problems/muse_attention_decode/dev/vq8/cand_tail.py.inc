

# ---------------------------------------------------------------------------------------------
# vq8 (q_len 1 and 8): both KV heads of a pair per CTA (128-byte K/V rows), one persistent CTA per SM
# over a stream-K split of (request, tile), a 4-warp cp.async producer that also transposes V, two
# consumer warpgroups on native FP8 WGMMA with P kept in registers, and a PDL combine that applies
# RMSNorm x sigmoid(gate). Source: problems/muse_attention_decode/dev/vq8. Everything above is the
# scored d98ae45d submission (its prepare-time portfolio picks the q_len-8 incumbent); the import-time
# lab below routes a shape to vq8 only where vq8 beats the incumbent (q_len 8: the best of FA3's default
# and the two-group math kernel, the portfolio's winners; q_len 1: the scored q1 path).
# ---------------------------------------------------------------------------------------------
_BASE_FN = kernel_fn

VQ8_CPP = r'''
#include <cstdint>
#include <vector>
void vq8_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,double,int64_t,int64_t,int64_t);
std::vector<int64_t> vq8_resources(int64_t);
'''
_VQ8_EXT = {}
_VQ8_PLAN = {}
_VQ8_OK = None


def _vq8_ext(clk=False):
    """clk=True: the phase-clock build (VQ8_CLK), lab only."""
    major, minor = torch.cuda.get_device_capability()
    arch = '90a' if major == 9 else f'{major}{minor}'
    key = (arch, clk)
    if key not in _VQ8_EXT:
        import hashlib
        tag = hashlib.sha256(VQ8_CUDA.encode()).hexdigest()[:12]
        _VQ8_EXT[key] = load_inline(name=f'vq8{"clk" if clk else ""}_{tag}_{arch}', cpp_sources=VQ8_CPP,
                                    cuda_sources=VQ8_CUDA, functions=['vq8_forward', 'vq8_resources'],
                                    extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{arch},code=sm_{arch}',
                                                       f'-DVQ8_CLK={int(clk)}'],
                                    verbose=False)
    return _VQ8_EXT[key]


def _vq8_supported(cfg):
    hq, hk = cfg["num_q_heads"], cfg["num_kv_heads"]
    return (cfg["q_len"] in (1, 8) and cfg["head_dim"] == 64 and hq == 8 * hk and hk % 2 == 0
            and cfg["batch"] * (hk // 2) <= 128)


def _vq8(data, stages=8, min_tiles=2, grid=None, clocks=None):
    cfg = data["config"]
    B, HQ, HK = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    device = qkv.device
    if torch.cuda.get_device_capability(device)[0] != 9:
        stages = 2  # sm_120 screen: 99 KB of shared memory per block
    G = grid or _num_sms(device)
    NS = B * (HK // 2)
    L = cfg["q_len"]
    nclk = (G + 16 * L * NS) * 64 if clocks is not None else 0  # up to 8 warps per combine row
    part = torch.empty(((G + NS) * 128 * 68 + 256 + nclk,), device=device, dtype=torch.float32)
    out = torch.empty((B, L, HQ * 64), device=device, dtype=torch.bfloat16)
    window = cfg["window"]
    invf = _inv_freq(cfg["rope_theta"], device) if window else None
    if clocks is not None:
        part[(G + NS) * 128 * 68 + 256:].zero_()
    _vq8_ext(clocks is not None).vq8_forward(
        qkv.data_ptr(), data["cache_loc"].data_ptr(), data["page_table"].data_ptr(), data["seq_lens"].data_ptr(),
        data["k_cache"].data_ptr(), data["v_cache"].data_ptr(), data["k_scale"].data_ptr(), data["v_scale"].data_ptr(),
        invf.data_ptr() if invf is not None else 0, part.data_ptr(), out.data_ptr(), B, L, HQ, HK, qkv.shape[-1],
        data["page_table"].shape[1], cfg["page_size"], window, 0, min_tiles, G, stages,
        cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634, 0, 0, torch.cuda.current_stream(device).cuda_stream)
    if clocks is not None:
        clocks.append(part[(G + NS) * 128 * 68 + 256:].view(torch.int64).view(G + 16 * L * NS, 32))
    return out


def _clock_summary(rows, G):
    """One line from _vq8(clocks=...) rows: times in us from the first CTA's entry, wait shares (vq8.cu maps)."""
    rows = rows.cpu()
    main, comb = rows[:G], rows[G:]
    comb = comb[comb[:, 0] != 0]
    act = main[main[:, 10] != -1]
    t0 = int(main[:, 0].min())
    us = lambda x: (x.double() - t0) / 1000.0  # noqa: E731
    med = lambda x: float(x.double().median())  # noqa: E731
    cyc_c, cyc_p = act[:, 5].double().clamp(min=1), act[:, 9].double().clamp(min=1)
    return ("ctas=%d tiles=%d-%d entry_max=%.2f sched=%.2f q_ready=%.2f first=%.2f/%.2f wg0_end=%.2f/%.2f "
            "wg1_end_max=%.2f prod_end_max=%.2f cwait=%.2f/%.2f pempty=%.2f pcp=%.2f comb_entry=%.2f comb_go=%.2f "
            "comb_end=%.2f cyc_per_tile=%.0f per-tile cyc qkw=%.0f smx=%.0f pvw=%.0f iss=%.0f pub=%.0f pis=%.0f" % (
                len(act), int(act[:, 10].min()), int(act[:, 10].max()), float(us(main[:, 0]).max()),
                med(us(act[:, 1]) - us(act[:, 0])), med(us(act[:, 15])), med(us(act[:, 2])), float(us(act[:, 2]).max()),
                med(us(act[:, 3])), float(us(act[:, 3]).max()), float(us(act[:, 13]).max()), float(us(act[:, 6]).max()),
                med(act[:, 4] / cyc_c), med(act[:, 11] / cyc_c), med(act[:, 7] / cyc_p), med(act[:, 8] / cyc_p),
                float(us(comb[:, 0]).min()), float(us(comb[:, 1]).min()), float(us(comb[:, 2]).max()),
                med(cyc_c / act[:, 10].double().clamp(min=1)),
                *[med(act[:, c].double() / act[:, 10].double().clamp(min=1)) for c in range(19, 25)]))


def _lab_input(B, window, ctx, seed, device, L=8):
    """Synthetic inputs with the scored layout (page 1, scattered pages, agentic lengths)."""
    g = torch.Generator(device=device).manual_seed(seed)
    HQ, HK = 16, 2
    if B > 1:
        seq = (ctx * (1 + 0.075 * (2 * torch.rand(B, generator=g, device=device) - 1))).long()
    else:
        seq = torch.full((1,), ctx, dtype=torch.long, device=device)
    longest = int(ctx * 1.075) + 1 if B > 1 else ctx
    max_pages = longest + L
    total = B * max_pages
    page_table = torch.randperm(total, generator=g, device=device).to(torch.int32).view(B, max_pages)
    k_cache = (torch.randn((total, 1, HK, 64), generator=g, device=device) * 0.5).to(torch.float8_e4m3fn)
    v_cache = torch.randn((total, 1, HK, 64), generator=g, device=device).to(torch.float8_e4m3fn)
    qkv = torch.randn((B, L, (2 * HQ + 2 * HK) * 64), generator=g, device=device).to(torch.bfloat16)
    pos = seq[:, None] + torch.arange(L, device=device)
    cache_loc = torch.gather(page_table.long(), 1, pos).reshape(-1)
    return {"qkv": qkv, "positions": pos.reshape(-1), "cache_loc": cache_loc,
            "cache_seqlens": (seq + L).to(torch.int32),
            "cu_seqlens_q": torch.arange(0, B * L + 1, L, dtype=torch.int32, device=device),
            "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache,
            "k_scale": torch.tensor(1.0, device=device), "v_scale": torch.tensor(1.0, device=device),
            "page_table": page_table, "seq_lens": seq.to(torch.int32),
            "config": {"batch": B, "q_len": L, "num_q_heads": HQ, "num_kv_heads": HK, "head_dim": 64, "page_size": 1,
                       "window": window, "rope_theta": 500000.0 if window else 0.0,
                       "qk_scale_factor": 5.4730064863875, "sm_scale": 0.125, "kv_scale": 1.0}}


def _own_reference(data):
    """FP32 attention of the step, attending the kernel's own new cache entries (already written)."""
    cfg = data["config"]
    B, L, HQ, HK = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    x = data["qkv"].float().view(B, L, -1)
    q = x[..., :HQ * 64].reshape(B, L, HQ, 64)
    gate = x[..., (HQ + 2 * HK) * 64:]
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5)
    if cfg["window"]:
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, 64, 2, device=q.device).float() / 64))
        ang = data["positions"].view(B, L).float()[..., None] * inv
        c, s = ang.cos()[:, :, None], ang.sin()[:, :, None]
        ev, od = q[..., 0::2], q[..., 1::2]
        q = torch.stack([ev * c - od * s, ev * s + od * c], -1).flatten(-2)
    q = q.to(torch.float8_e4m3fn).float() * cfg["qk_scale_factor"]
    kf = data["k_cache"].view(-1, HK, 64).float() * data["k_scale"]
    vf = data["v_cache"].view(-1, HK, 64).float() * data["v_scale"]
    out = torch.empty((B, L, HQ, 64), device=q.device)
    for b in range(B):
        seq = int(data["seq_lens"][b])
        rows = data["page_table"][b, :seq + L].long()
        kb, vb = kf[rows], vf[rows]
        for j in range(L):
            p = seq + j
            lo = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            att = torch.einsum("kgd,tkd->kgt", q[b, j].view(HK, HQ // HK, 64), kb[lo:p + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("kgt,tkd->kgd", att.softmax(-1), vb[lo:p + 1]).reshape(HQ, 64)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + 1e-5) * torch.sigmoid(gate.view(B, L, HQ, 64))
    return out.reshape(B, L, HQ * 64)


def _graph_time(fn, flush, reps=15):
    for _ in range(2):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(reps):
        flush.sum()  # a read larger than L2, as the judge does
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    del g
    vals.sort()
    return vals[len(vals) // 2]


def _vq8_lab():
    global _VQ8_OK
    device = torch.device("cuda", torch.cuda.current_device())
    import time
    t0 = time.monotonic()
    ext = _vq8_ext()
    _vq8_ext(True)
    print("VQ8_COMPILE_S %.1f" % (time.monotonic() - t0), flush=True)
    for st in (4, 6, 8):
        print("VQ8_RES st=%d regs,local,static_smem,dyn_smem,ctas_per_sm=%s" % (st, ext.vq8_resources(st)), flush=True)
    errs = []
    for L in (8, 1):
        for window in (0, 2048):
            for (B, ctx) in ((2, 3000), (3, 200)):
                d = _lab_input(B, window, ctx, 11 + B + window, device, L)
                out = _vq8(d).float()
                torch.cuda.synchronize()
                ref = _own_reference(d)
                errs.append(float((out - ref).norm() / ref.norm()))
                if errs[-1] >= 0.12:  # the error's pattern over tokens, heads and dims, for the next fix
                    B_, HQ_ = d["config"]["batch"], d["config"]["num_q_heads"]
                    e = (out - ref).view(B_, L, HQ_, 64).float()
                    r = ref.view(B_, L, HQ_, 64).float()
                    def rel(dim):
                        return ",".join("%.2f" % v for v in (e.pow(2).sum(dim) / r.pow(2).sum(dim)).sqrt().flatten().tolist())
                    cos = float((out * ref).sum() / (out.norm() * ref.norm()))
                    print("VQ8_GATE_DIAG L=%d W=%d B=%d cos=%.4f tok=[%s] head=[%s] dim16=[%s] finite=%d" % (
                        L, window, B, cos, rel((0, 2, 3)), rel((0, 1, 3)),
                        ",".join("%.2f" % v for v in (e.pow(2).view(B_, L, HQ_, 4, 16).sum((0, 1, 2, 4))
                                                       / r.pow(2).view(B_, L, HQ_, 4, 16).sum((0, 1, 2, 4))).sqrt().tolist()),
                        int(torch.isfinite(out).all())), flush=True)
    _VQ8_OK = max(errs) < 0.12
    print("VQ8_GATE ok=%d errs=%s" % (_VQ8_OK, ",".join("%.4f" % e for e in errs)), flush=True)
    if not _VQ8_OK:
        return
    flush = torch.empty(1 << 28, dtype=torch.int64, device=device)  # 2 GiB, read
    configs = ((8, 2), (6, 2), (4, 2), (8, 4), (8, 8), (8, 1), (6, 1), (4, 1))
    for L in (8, 1):
        for B in (1, 16, 32):
            for window in (2048, 0):
                lab = _lab_input(B, window, 80000, 100 + B + window, device, L)
                if L == 1:
                    base = _graph_time(lambda x=lab: _BASE_FN(x), flush)
                else:
                    base = min(_graph_time(lambda x=lab: _fa3_verify(x), flush),
                               _graph_time(lambda x=lab: _vmath_twog_verify(x, 1), flush),
                               _graph_time(lambda x=lab: _vmath_twog_verify(x, 2), flush))
                times = {}
                for st, mt in configs:
                    times[(st, mt)] = _graph_time(lambda x=lab, a=st, b=mt: _vq8(x, stages=a, min_tiles=b), flush)
                best = min(times, key=times.get)
                _VQ8_PLAN[(B, L, bool(window))] = best if times[best] < base else None
                print("VQ8_LAB L=%d B=%d W=%d base=%.1f %s pick=%s" % (
                    L, B, window, base,
                    " ".join("st%dmt%d=%.1f" % (k[0], k[1], v) for k, v in times.items()),
                    "st%dmt%d" % best if times[best] < base else "base"), flush=True)
                # Phase clocks of the best vq8 setting (cold L2, as timed).
                rows = []
                _vq8(lab, stages=best[0], min_tiles=best[1])
                flush.sum()
                _vq8(lab, stages=best[0], min_tiles=best[1], clocks=rows)
                torch.cuda.synchronize()
                print("VQ8_CLK L=%d B=%d W=%d st%dmt%d %s" % (L, B, window, best[0], best[1],
                                                           _clock_summary(rows[0], _num_sms(device))), flush=True)
                lab = None
                torch.cuda.empty_cache()
    print("VQ8_LAB_S %.1f" % (time.monotonic() - t0), flush=True)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    hopper = torch.cuda.get_device_capability(data["qkv"].device)[0] == 9
    if _vq8_supported(cfg) and (not hopper or _VQ8_OK):
        # Hopper: vq8 only for shapes where the lab measured it faster than the incumbent dispatch.
        # Off Hopper (local screen) every supported shape runs vq8, so the tests exercise it.
        choice = _VQ8_PLAN.get((cfg["batch"], cfg["q_len"], bool(cfg["window"])), None if hopper else (8, 2))
        if choice is not None:
            return _vq8(data, stages=choice[0], min_tiles=choice[1])
    return _BASE_FN(data)


if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
    try:
        _vq8_lab()
    except Exception as exc:  # any lab failure keeps the incumbent dispatch everywhere
        _VQ8_OK = False
        _VQ8_PLAN.clear()
        print("VQ8_LAB_ERROR %s" % str(exc).splitlines()[0][:200], flush=True)
