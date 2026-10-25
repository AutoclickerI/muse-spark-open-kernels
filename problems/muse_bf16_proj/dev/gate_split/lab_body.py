# Gate split chain (P7), paired on identical inputs at every q_len 1 shape, from the layer input to O's consumer:
#   ctrl   S10 + gemv_bal: qkv_gate (bal 64/7, 2304 rows) -> sum+cast -> attention (S10 ext, gated) -> O (S10 V9)
#          -> b1 -> sink
#   split  qkv3 (bal 64/10, 1280 rows) -> sum+cast -> attention (patched ext: a 1280-wide row = ungated; the layer
#          ext's attn_main / attn_combine release dependents, attn_combine PDL-waits) -> _gate_split_kernel (gate
#          rows' GEMV before griddepcontrol.wait; BN 64, S 8, 4 stages) -> O -> b1 -> sink. gsS = S gate splits.
GS_ARMS = (('split', (64, 8, 4)), ('gs4', (64, 4, 4)), ('gs16', (64, 16, 4)))
GS_SHAPES = ((1, 2048), (1, 0), (16, 2048), (32, 2048), (32, 0), (16, 0))  # (batch, window), by score weight


class GsChain:
    def __init__(self, v, attn, x, gs):
        self.v, self.attn, self.x, self.gs = v, attn, x, gs
        T = x.shape[0]
        self.sink = torch.empty(T, device=x.device, dtype=torch.float32)
        if gs is not None:  # prepare-time partials and counters (the kernel writes them before its wait)
            BN, S, _ = gs
            self.P = torch.empty((S, T, 1024), device=x.device, dtype=torch.float32)
            self.cnt = torch.zeros(1024 // BN, device=x.device, dtype=torch.int32)

    def __call__(self):
        v, x, T = self.v, self.x, self.x.shape[0]
        if self.gs is None:
            P, S = ro_bal(x, v['qkv'], 64, 7)
            a = self.attn(ro_sumcast(P, S))
        else:
            P, S = ro_bal(x, v['qkv3'], 64, 10)
            y = self.attn(ro_sumcast(P, S), gs=True)
            BN, S2, NS = self.gs
            a = torch.empty((T, 1024), device=x.device, dtype=torch.bfloat16)
            _gate_split_kernel[(1024 // BN, S2)](x, v['gate'], self.P, self.cnt, y, a, T, x.stride(0), 8192, S2, N=1024,
                                                 BN=BN, BK=128, TB=max(16, triton.next_power_of_2(T)), NS=NS,
                                                 num_warps=4, launch_pdl=True)
        o, so = ro_s10(a, v['o'], (64, 512, 128))
        h, _ = ro_boundary(v, o, so, 8192, False)
        _ro_sink[(T,)](h, self.sink, N=8192, NP=8192, launch_pdl=True, num_warps=8)
        return a, h


def _gs_weights(weights, dev):
    """The proj host's dense weights, or (other hosts) synthetic ones at the real shapes: randn / sqrt(k) BF16."""
    if 'qkv_gate' in weights:
        return weights
    g = torch.Generator(device=dev).manual_seed(1280)
    r = lambda n, k, dt=torch.bfloat16: (torch.randn((n, k), device=dev, generator=g) / k ** .5).to(dt)  # noqa: E731
    return dict(qkv_gate=r(2304, 8192), o=r(8192, 1024), pooler=r(4096, 8192), unpooler=r(8192, 4096),
                router=r(256, 8192, torch.float32))


def lab(deadline, weights):
    t0 = time.monotonic()
    dev = torch.device('cuda')
    weights = _gs_weights(weights, dev)
    gen = torch.Generator(device=dev).manual_seed(13)
    lines = ['META paired AB/BA x24, lab_cold_l2 per replay; deltas in us = (paired ratio - 1) x ctrl median; e = max '
             'rel L2 of the gated attention output vs ctrl. ctrl: qkv bal64/7 (2304) -> sumcast -> attn (S10) -> O -> b1 '
             '-> sink; split/gsS: qkv3 bal64/10 (1280) -> sumcast -> attn (patched, ungated) -> gate kernel (64/S/4) -> '
             'O -> b1 -> sink.']
    for b, w in GS_SHAPES:
        if time.monotonic() > deadline - 20:
            lines.append(f'SKIP GS b={b} w={w}')
            continue
        try:
            v = _ro_vectors(weights, dev, gen, b)
            v['qkv3'], v['gate'] = v['qkv'][:1280], v['qkv'][1280:]
            attn = RoAttention(b, w, dev)
            x = (torch.randn((b, 8192), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            arms = [('ctrl', GsChain(v, attn, x, None), None), ('AA', GsChain(v, attn, x, None), None)]
            arms += [(name, GsChain(v, attn, x, cfg), None) for name, cfg in GS_ARMS]
            got, err = _ro_case(arms)
            _, ns, cs, _, _, _ = attn.plan(False)
            lines.append(f'GS b={b} w={w} {attn.kind} ns/cs={ns}/{cs} ctrl={got["ctrl"][0]:.2f}us AA={got["AA"][1]:.4f} | '
                         + _ro_fmt(got, [n for n, _ in GS_ARMS], 'ctrl') + f' | e={err:.0e}')
        except Exception as exc:
            lines.append(f'ERROR GS b={b} w={w} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    if RO_REJECTED:
        lines.append('REJECTED ' + ' '.join(sorted(set(RO_REJECTED))))
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
