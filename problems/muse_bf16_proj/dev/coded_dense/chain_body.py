

# ---- coded GEMVs in their S8/S10 chains (paired, lab_cold_l2 per replay) ----


class CCPool:
    """O producer (cuBLAS F.linear stand-in, or coded) -> boundary_cl b1 ('cl', 8, 256, 1) -> pool_router producer
    -> route_cl ('cl', 8, 256, 0) -> PDL consumer (S10's decode chain from attention to the experts)."""

    def __init__(self, t, v, attn, o_prod, prod):
        self.t, self.v, self.attn, self.o_prod, self.prod = t, v, attn, o_prod, prod
        self.sink = torch.empty(t, device=attn.device, dtype=torch.float32)

    def __call__(self):
        t, v, ext, st = self.t, self.v, gp_ext(), torch.cuda.current_stream().cuda_stream
        o, so = self.o_prod(self.attn)
        h = torch.empty((t, 8192), device=o.device, dtype=torch.bfloat16)
        sout = torch.empty_like(v['stream'])
        ext.boundary_cl(o.data_ptr(), int(o.dtype == torch.float32), so, o.stride(0), o.stride(1), 8192,
                        v['stream'].data_ptr(), sout.data_ptr(), h.data_ptr(), 0, v['alpha1'].data_ptr(),
                        v['beta1'].data_ptr(), v['pre_ffn_w'].data_ptr(), t, 8, 256, 1, st)
        p, s = self.prod(h)
        x = torch.empty((t, 4096), device=o.device, dtype=torch.bfloat16)
        ids = torch.empty((t, 8), device=o.device, dtype=torch.int32)
        wts = torch.empty((t, 8), device=o.device, dtype=torch.float32)
        lids = torch.empty_like(ids)
        ext.route_cl(p.data_ptr(), 1, s, p.stride(0), p.stride(1), v['bias'].data_ptr(), v['pre_expert_w'].data_ptr(),
                     v['lstart'].data_ptr(), x.data_ptr(), ids.data_ptr(), wts.data_ptr(), lids.data_ptr(), 0, 0, t, 8,
                     256, st)
        _gp_consume[(t,)](x, wts, self.sink, NA=4096, NB_=8, launch_pdl=True, num_warps=4)
        return x, ids, wts, sout


# Standalone H200 winners (coded_gemv receipt) per (t, role): (NW, RT, PIPE, S, V, C); each runs with prew 1 and 0.
CC_CFG = __CC_CFG__
CC_INSTANCES = tuple(sorted({(1 if t <= 8 else 2 if t <= 16 else 4, rt, pipe, v)
                             for (t, _), cfgs in CC_CFG.items() for _, rt, pipe, _, v, _ in cfgs}))
CC_S8 = {1: (32, 1024, 128, False), 8: (32, 1024, 128, False), 16: (32, 1024, 128, False), 32: (64, 512, 128, False)}
CC_QKV_S8 = {1: 'cublas', 8: 'cublas', 16: (32, 1024, 128, False), 32: 'cublas'}  # S8 BEST6 plan (a5b2f55b)
CC_BAL = {'qkv': 7, 'pool_router': 6, 'unpooler': 2, 'unpool_slice': 16, 'o': 2}


def cc_arms(t, role, x_w, c):
    out = []
    for cfg in CC_CFG.get((t, role), ()):
        NW, RT, PIPE, S, V, C = cfg
        for prew in (1, 0):
            out.append((f'V{V}:{NW}/{RT}/{PIPE}/{S}/{C}{"p" if prew else "w"}',
                        lambda x, NW=NW, RT=RT, PIPE=PIPE, S=S, V=V, C=C, prew=prew: cd_gemv(
                            x, c, S, NW=NW, RT=RT, PIPE=PIPE, prew=prew, V=V, C=C)))
    return out


def lab(deadline, weights):
    t0 = time.monotonic()
    weights, wkind = cl_weights(weights)
    dev = weights['o'].device
    gen = torch.Generator(device=dev).manual_seed(5)
    cd_module(CC_INSTANCES)
    gp_ext()
    v0 = _gp_vectors(weights, dev, gen, 1)
    coded = dict(pool_router=cd_encode(v0['pool_router']), qkv=cd_encode(v0['qkv']), unpooler=cd_encode(v0['unpooler']),
                 unpool_slice=cd_encode(v0['unpooler'][:1024].contiguous()), o=cd_encode(v0['o']))
    lines = [f'META {wkind} weights; paired chains, lab_cold_l2 per replay, r = paired median vs the first arm (control), AA = control vs '
             'itself; Vv:NW/RT/PIPE/S/C p|w = coded GEMV step v with its first PIPE blocks loaded before (p) or after (w) '
             'griddepcontrol.wait. POOL: O (cuBLAS) -> b1 -> pool_router -> route -> sink; POOLO: the same with O coded '
             'too (vs S8 O stream/cuBLAS). QKV: qkv -> sum+cast -> sink. UNPOOL: unpooler (slice at t<=8) -> b2 -> sink.']
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))  # noqa: E731
    for t in (1, 8, 16, 32):
        if time.monotonic() > deadline - 25:
            lines.append(f'SKIP t={t}')
            continue
        v = _gp_vectors(weights, dev, gen, t)
        attn = (torch.randn((t, 1024), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
        cub_o = lambda x: (F.linear(x, v['o']).unsqueeze(0), 1)  # noqa: E731
        try:
            w = v['pool_router']
            s8 = lambda x: gp_s8(x, w, CC_S8[t])  # noqa: E731
            arms = [('s8', s8), ('s8b', s8), ('bal', lambda x: gp_run(x, w, 64, CC_BAL['pool_router'], 0, 0))]
            arms += cc_arms(t, 'pool_router', w, coded['pool_router'])
            _gp_case(lines, 'POOL', t, arms, lambda prod: CCPool(t, v, attn, cub_o, prod),
                     check=lambda o, r: '' if torch.equal(o[1], r[1]) and rel(o[0], r[0]) < 1e-2 else f'ids={torch.equal(o[1], r[1])},x={rel(o[0], r[0]):.1e}')
            best_pool = CC_CFG.get((t, 'pool_router'), [None])[0]
            if best_pool is not None and CC_CFG.get((t, 'o')):
                NW, RT, PIPE, S, V, C = best_pool
                pr = lambda x: cd_gemv(x, coded['pool_router'], S, NW=NW, RT=RT, PIPE=PIPE, prew=1, V=V, C=C)  # noqa: E731
                arms = [('s8', None), ('s8b', None)] + [(n, f) for n, f in cc_arms(t, 'o', None, coded['o'])]
                chains = [CCPool(t, v, attn, cub_o, s8), CCPool(t, v, attn, cub_o, s8)]
                chains += [CCPool(t, v, attn, f, pr) for _, f in arms[2:]]
                graphs = [_gp_graph(ch)[0] for ch in chains]
                res = _gp_paired(graphs, 24)
                lines.append(f'POOLO t={t} s8={res[0][0]:.2f}us AA={res[1][1]:.4f} ' +
                             ' '.join(f'[o:{n} {u:.2f}us r={r:.4f}]' for (n, _), (u, r) in zip(arms[2:], res[2:])))
                del graphs, chains
        except Exception as exc:
            lines.append(f'ERROR POOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        try:
            wq = v['qkv']
            x = (torch.randn((t, 8192), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            ctrl = CC_QKV_S8[t]
            s8 = (lambda x: (F.linear(x, wq).unsqueeze(0), 1)) if ctrl == 'cublas' else (lambda x: gp_s8(x, wq, ctrl))
            arms = [('s8', s8), ('s8b', s8), ('bal', lambda x: gp_run(x, wq, 64, CC_BAL['qkv'], 0, 0))]
            arms += cc_arms(t, 'qkv', wq, coded['qkv'])
            _gp_case(lines, 'QKV', t, arms, lambda prod: GPQkv(t, x, prod, 2304),
                     check=lambda o, r: '' if rel(o[0], r[0]) < 1e-2 else f'y={rel(o[0], r[0]):.1e}')
        except Exception as exc:
            lines.append(f'ERROR QKV t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        try:
            role = 'unpool_slice' if t <= 8 else 'unpooler'
            wu = v['unpooler'][:1024] if t <= 8 else v['unpooler']
            xu = (torch.randn((t, 4096), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            if t <= 8:
                ctrl = lambda x: gp_s8(x, wu, (64, 512, 128, False))  # noqa: E731
            else:
                ctrl = lambda x: (F.linear(x, wu).unsqueeze(0), 1)  # noqa: E731
            arms = [('s8', ctrl), ('s8b', ctrl), ('bal', lambda x: gp_run(x, wu, 64, CC_BAL[role], 0, 0))]
            arms += cc_arms(t, role, wu, coded[role])
            _gp_case(lines, 'UNPOOL', t, arms, lambda prod: GPUnpool(t, xu, v, prod),
                     check=lambda o, r: '' if rel(o[1], r[1]) < 1e-2 else f'sout={rel(o[1], r[1]):.1e}')
        except Exception as exc:
            lines.append(f'ERROR UNPOOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}')
        torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
