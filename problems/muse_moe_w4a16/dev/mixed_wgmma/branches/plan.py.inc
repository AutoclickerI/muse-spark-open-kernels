import functools
import hashlib
import statistics
import time
import torch
from torch.utils.cpp_extension import load_inline

BR_DEADLINE = float('inf')
BR_OPTS = {}


class BRBudget(RuntimeError):
    pass


def br_need(seconds, where):
    if BR_DEADLINE - time.monotonic() < seconds:
        raise BRBudget(where)


def br_compile(name, cpp, cuda, functions):
    br_need(30., name + ' compile')
    arch = '90a' if torch.cuda.get_device_capability() == (9, 0) else '120'
    return load_inline(name=name + '_' + hashlib.sha256(cuda.encode()).hexdigest()[:12], cpp_sources=cpp, cuda_sources=cuda,
                       functions=functions, no_implicit_headers=True, with_pytorch_error_handling=False,
                       extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                       verbose=False)


BR_FUNCS = ['br_init', 'br_info', 'br_run', 'br_route', 'br_finish']


@functools.lru_cache(None)
def br_hh_module():
    mod = br_compile('brhh', BR_HH_CPP, BR_HH_CUDA, BR_FUNCS + ['pmz_init', 'pmz_smem', 'pmz_resident', 'pmz_run'])
    mod.br_init()
    BR_OPTS['pmz'] = mod.pmz_init()
    return mod


@functools.lru_cache(None)
def br_s6_module():
    mod = br_compile('brs6', BR_S6_CPP, BR_S6_CUDA, BR_FUNCS)
    mod.br_init()
    return mod


@functools.lru_cache(None)
def br_u16_module():
    # S6's H_H CUDA (the heavy WGMMA reads prepare-widened u16 scale words): measurement-only reference
    mod = br_compile('bru16', M128_HH_PM_CPP, BR_U16_CUDA, M128_HH_PM_FUNCTIONS)
    mod.idist_init()
    return mod


def br_modules():
    m128_hh_pm_module()
    br_hh_module()
    if BR_SET == 'tune':
        return
    if BR_M == 128:
        br_u16_module()
    else:
        br_s6_module()


def br_prepare(weights):
    hh = hh_prepare(weights)
    dev = hh['base']['w13'].device
    hh.update(side=torch.cuda.Stream(device=dev), side_hi=torch.cuda.Stream(device=dev, priority=-1), fork_ev=torch.cuda.Event(), join_ev=torch.cuda.Event(), mod=br_hh_module(),
              sms=torch.cuda.get_device_properties(dev).multi_processor_count)
    if BR_SET == 'tune':
        pass
    elif BR_M == 128:
        hh['u16_sf'] = {k: ((hh['base'][k].to(torch.int32) << 4) + 0x7000).to(torch.int16) for k in ('w13_sf', 'w2_sf')}
    else:
        hh['s6'] = wi_prepare(weights)
    return hh


# Heavy kernels on H_H's buffers (gemm.cu.inc kinds; u8 scale codes): 0/1 W13/W2 N32 on the heavy-only
# (count>16) T32 list; 4/8 W13/W2 N32 groups with >16 rows, 5/9 N16 groups with <=16 rows (heavy remainders);
# 12 W2 N16 for experts with 8<count<=16 (T8 list). 'n' = NT1 for count<=16, 'n8' = W2 NT1 for count<=8.
# Spec: (W13 side, W13 main, W2 side, W2 main, mode); seq = one stream per phase, phase = fork/join per
# phase, full = fork once (side W13 -> side W2 || main W13 -> main W2), join before the finalizer.
BR_ARMS = {
    'HH': None,        # H_H on u8 codes: M128 = exact S7; M256 = the same policy (ec44b322 H16g1056)
    'S6': 's6',        # M256: exact S6 mixed_parallel
    'U16': 'u16',      # M128: exact u16-word H_H (770809cf / S6): the cost of S7's in-register word
    'R_s6': 's6R',     # M256: 110599ba winner R_full on S6's buffers
    'HH_u8': 'u8',     # H_H with its heavy WGMMA replaced by this lab's typed-smem kernel
    'N_phase': (('n',), (0,), ('n',), (1,), 'phase'),
    'N_full': (('n',), (0,), ('n',), (1,), 'full'),
    'NR_full': (('n', 5), (4,), ('n', 9), (8,), 'full'),
    'NRm_full': (('n',), (4, 5), ('n',), (8, 9), 'full'),
    'B8_full': (('n',), (0,), ('n8', 12), (1,), 'full'),
    'B8R_full': (('n', 5), (4,), ('n8', 12, 9), (8,), 'full'),
}
BR_GRID = 1056 if BR_M == 128 else 2112
# N_full tuning (control N_full = nf_cell.py): heavy grid, NT1 persistent-grid fraction, a high-priority side stream,
# heavy on the side stream instead of NT1.
BR_TUNE = {
    'N_g' + str(BR_GRID // 2): (('n',), (0,), ('n',), (1,), 'full', dict(grid=BR_GRID // 2)),
    'N_g' + str(BR_GRID * 2): (('n',), (0,), ('n',), (1,), 'full', dict(grid=BR_GRID * 2)),
    'N_p50': (('n',), (0,), ('n',), (1,), 'full', dict(pg=.5)),
    'N_p75': (('n',), (0,), ('n',), (1,), 'full', dict(pg=.75)),
    'N_hi': (('n',), (0,), ('n',), (1,), 'full', dict(hi=True)),
    'N_swap': ((0,), ('n',), (1,), ('n',), 'full'),
    'N_swap_hi': ((0,), ('n',), (1,), ('n',), 'full', dict(hi=True)),
    'N_p50_hi': (('n',), (0,), ('n',), (1,), 'full', dict(pg=.5, hi=True)),
}
BR_ARMS.update(BR_TUNE)
BR_ONLY = {'S6': 256, 'R_s6': 256, 'U16': 128}
if BR_SET == 'tune':
    BR_CONTROL = 'N_full'
    BR_NAMES = [BR_CONTROL] + list(BR_TUNE)
else:
    BR_CONTROL = 'HH' if BR_M == 128 else 'S6'
    BR_NAMES = [BR_CONTROL] + [a for a in BR_ARMS if a != BR_CONTROL and a not in BR_TUNE and BR_ONLY.get(a, BR_M) == BR_M]


def br_cur():
    return torch.cuda.current_stream().cuda_stream


def br_fork(hh, side, main, key='side'):
    parent = torch.cuda.current_stream()
    hh['fork_ev'].record(parent)
    hh[key].wait_event(hh['fork_ev'])
    with torch.cuda.stream(hh[key]):
        side(hh[key].cuda_stream)
        hh['join_ev'].record()
    main(parent.cuda_stream)
    parent.wait_event(hh['join_ev'])


def br_s6_plan(data, hh, spec):
    bank = hh['s6']
    plan = wi_plan(data, bank, 6, 2112)
    v13, v2 = plan['v13'], plan['v2']
    plan = dict(plan, out=v2[19])
    if spec == 's6':
        return plan
    mod = br_s6_module()
    m, start = plan['M'], plan['start']
    q13 = [bank['w13_rec'], bank['w13_rec_sf']] + [v13[i] for i in (2, 3, 4, 5, 6, 7, 8, 14, 9, 15, 16)]
    q2 = [bank['w2_rec'], bank['w2_rec_sf']] + [v2[i] for i in (2, 3, 4, 5, 6, 7, 8, 14, 9, 15, 16)]
    p13, p2 = [t.data_ptr() for t in q13], [t.data_ptr() for t in q2]
    rp = [v2[i].data_ptr() for i in (11, 4, 5, 6, 7, 8, 13, 7, 7)]
    fp = [v2[i].data_ptr() for i in (5, 11, 12, 13, 15, 16, 19)]
    owners = (q13, q2, v13, v2, plan)

    def route():
        _ = owners
        mod.br_route(rp, m, start, 0, br_cur())

    def body():
        _ = owners
        br_fork(hh, lambda s: (mod.br_run(p13, 5, 2112, m, s), mod.br_run(p2, 9, 2112, m, s)),
                lambda s: (mod.br_run(p13, 4, 2112, m, s), mod.br_run(p2, 8, 2112, m, s)))

    def finish():
        _ = owners
        mod.br_finish(fp, m, start, br_cur())
        return v2[19]
    return dict(plan, steps=(route, body, finish))


def br_plan(data, hh, arm):
    spec = BR_ARMS[arm]
    if spec in ('s6', 's6R'):
        return br_s6_plan(data, hh, spec)
    base, native = hh['base'], hh['native']
    plan = m128_hh_pm_plan(data, base, native)
    if spec is None:
        return plan
    mod, hmod = hh['mod'], m128_hh_pm_module()
    v13, tasks = plan['v13'], plan['tasks']
    t8 = torch.empty_like(tasks[0])
    x, c, off, order, slots, act = v13[3], v13[4], v13[5], v13[6], v13[8], v13[9]
    ids, tw, pos, y, ss, out = v13[11], v13[12], v13[13], v13[15], v13[16], v13[19]
    m = x.shape[0]
    start = data['config'].get('local_start', 0)
    h13 = [base['w13'], base['w13_sf'], base['w13_gs'], x, c, off, order, tasks[1], slots, base['pnw'], act, y, ss]
    h2 = [base['w2'], base['w2_sf'], base['w2_gs'], act, c, off, order, tasks[1], slots, base['pnw'], act, y, ss]
    h28 = h2[:7] + [t8] + h2[8:]
    nv = [x, base['pnw'], native['w13p'], native['w2p'], base['w13_gs'], base['w2_gs'], c, off, slots, act, y, ss]
    p13, p2, p28, pn = ([t.data_ptr() for t in v] for v in (h13, h2, h28, nv))
    rp = [t.data_ptr() for t in (ids, c, off, order, tasks[0], slots, pos, tasks[1], t8)]
    fp = [t.data_ptr() for t in (off, ids, tw, pos, y, ss, out)]
    owners = (plan, h13, h2, h28, nv, t8)
    plan = dict(plan, tasks=list(tasks) + [t8])
    ostages = 2
    while ostages > 1 and hmod.pm_nt_smem(ostages, m) > M128_HH_ISSUE_OPTS['pm_nt']:
        ostages -= 1

    def hh_small(first):
        hmod.pm_nt_run(pn, first, 1, m, ostages, native['grid'] * hmod.pm_nt_resident(ostages, m, first, 1), br_cur())

    if spec in ('u16', 'u8'):
        if spec == 'u16':
            umod = br_u16_module()
            sf = hh['u16_sf']
            vu = [[base['w13'], sf['w13_sf']] + list(plan['v13'][2:]), [base['w2'], sf['w2_sf']] + list(plan['v2'][2:])]
            for v in vu:
                v[7] = tasks[1]
            qu = [[t.data_ptr() for t in v] for v in vu]
            cap = v13[9].shape[0]
            owners = owners + (vu,)

        def phase(first):
            def go():
                _ = owners
                hh_small(first)
                if spec == 'u16':
                    umod.idist_run(qu[0] if first else qu[1], first, 1, 1056, start, 1, 0, m, cap, br_cur())
                else:
                    mod.br_run(p13 if first else p2, 0 if first else 1, BR_GRID, m, br_cur())
            return go
        s = plan['steps']
        return dict(plan, steps=(s[0], phase(1), phase(0), s[3]))

    stages = 2
    while stages > 1 and mod.pmz_smem(stages, m) > BR_OPTS['pmz']:
        stages -= 1
    s13, k13, s2, k2, mode = spec[:5]
    opts = spec[5] if len(spec) > 5 else {}
    grids = {f: max(1, int(hh['sms'] * mod.pmz_resident(stages, m, f) * opts.get('pg', 1.))) for f in (0, 1)}
    hgrid = opts.get('grid', BR_GRID)
    skey = 'side_hi' if opts.get('hi') else 'side'

    def run(first, kinds, stream):
        for k in kinds:
            if k in ('n', 'n8'):
                mod.pmz_run(pn, first, 16 if k == 'n' else 8, m, stages, grids[first], stream)
            else:
                mod.br_run(p28 if k == 12 else p13 if first else p2, k, hgrid, m, stream)

    def route():
        _ = owners
        mod.br_route(rp, m, start, 1, br_cur())

    def finish():
        _ = owners
        mod.br_finish(fp, m, start, br_cur())
        return out

    if mode == 'full':
        def body():
            _ = owners
            br_fork(hh, lambda st: (run(1, s13, st), run(0, s2, st)), lambda st: (run(1, k13, st), run(0, k2, st)), skey)
        steps = (route, body, finish)
    else:
        def phase(first, side, main):
            def go():
                _ = owners
                if mode == 'seq':
                    run(first, side + main, br_cur())
                else:
                    br_fork(hh, lambda st: run(first, side, st), lambda st: run(first, main, st))
            return go
        steps = (route, phase(1, s13, k13), phase(0, s2, k2), finish)
    return dict(plan, steps=steps)


def br_call(data, hh, arm):
    for fn in br_plan(data, hh, arm)['steps']:
        out = fn()
    return out


def br_error(out, want):
    if not bool(torch.isfinite(out).all()):
        return float('inf'), float('inf')
    diff = out.float() - want.float()
    den = want.float().norm(dim=-1)
    return float(diff.norm() / want.float().norm().clamp_min(1e-12)), float((diff.norm(dim=-1) / den.clamp_min(den.mean() * .1 + 1e-12)).max())


def br_scratch(plan):
    unique = {}
    for v in (plan['v13'], plan['v2']):
        for i in (4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19):
            unique[v[i].data_ptr()] = v[i]
    for t in plan.get('tasks', []):
        unique[t.data_ptr()] = t
    return list(unique.values())
