"""Branch lab at M=128 or 256: every arm vs the FP32 oracle (TASK gates) with poisoned scratch, route transitions
(none/hot/remainder/single pair) through one captured graph, nearby M with a nonzero local_start; then the lab
itself under a deadline. Usage: check.py M [seeds] [lab_seconds] [tune]."""
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
R = H.parents[4]
_flush = []


def lab_cold_l2():
    if not _flush:
        _flush.append(torch.empty(1 << 30, dtype=torch.uint8, device='cuda'))
    _flush[0].sum()


def main():
    M = int(sys.argv[1])
    p = R / (f'prof/lab/moe_br_m{M}.py' if len(sys.argv) < 5 else f'prof/lab/moe_nf_tune_m{M}.py')
    m = runpy.run_path(str(p), init_globals=dict(lab_cold_l2=lab_cold_l2))
    arms = [a for a in m['BR_NAMES'] if a not in m.get('BR_PROBES', ())]  # probes are timing-only
    ctl = m['BR_CONTROL']
    checks = bit = 0
    worst = [0., 0.]
    gen = torch.Generator(device='cuda').manual_seed(4242)
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    for seed in range(seeds):
        w = m['dc_weights'](5353 + 7919 * seed, 'cuda')
        bank = m['br_prepare'](w)
        real = m['dc_fixtures'](w, M)
        base = real[0]
        none = dict(base, topk_ids=torch.full_like(base['topk_ids'], -1))
        hot = torch.full_like(base['topk_ids'], -1); hot[:, 0] = 0          # one expert with M rows
        rem = torch.full_like(hot, -1)
        for k, (e, n) in enumerate(((3, 40), (5, 7), (9, 16), (11, 17), (12, 33), (20, 9), (21, 8), (22, 12))):
            rem[:n, k] = e
        one = torch.full_like(hot, -1); one[17, 3] = 31
        seq = [none, real[0], none, dict(base, topk_ids=hot), dict(base, topk_ids=rem), real[1], none, dict(base, topk_ids=one), real[2], none]
        wants = m['dc_oracles'](seq, w)
        static = {k: v.clone() if torch.is_tensor(v) else v for k, v in base.items()}
        plans, graphs = {}, {}
        for a in arms:
            plans[a] = m['br_plan'](static, bank, a)
            for f in plans[a]['steps']:
                f()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in plans[a]['steps']:
                    f()
            graphs[a] = g
        for i, d in enumerate(seq):
            for k in ('hidden_states', 'topk_ids', 'topk_weights'):
                static[k].copy_(d[k])
            outs = {}
            for a in arms:
                for t in m['br_scratch'](plans[a]):
                    t.view(torch.uint8).random_(0, 256, generator=gen)
                graphs[a].replay()
                torch.cuda.synchronize()
                outs[a] = plans[a]['out'].clone()
                e, r = m['br_error'](outs[a], wants[i]) if d is not none else (0., 0.)
                assert torch.isfinite(outs[a]).all(), (seed, i, a)
                if d is none:
                    assert bool((outs[a].view(torch.int16) == 0).all()), (seed, i, a, 'none must be +0')
                assert e <= .015 and r <= .075, (seed, i, a, e, r)
                worst = [max(worst[0], e), max(worst[1], r)]
                checks += 1
                bit += bool(torch.equal(outs[a], outs[ctl]))
        print('SEED', seed, 'checks', checks, 'bit_equal_to', ctl, bit, 'worst', worst, flush=True)
    for size in ((64, 127, 128) if M == 128 else (192, 255, 256)):
        d = m['dc_fixtures'](w, size)[0]
        want = m['dc_oracles']([d], w)[0]
        dd = dict(d, topk_ids=torch.where(d['topk_ids'] >= 0, d['topk_ids'] + 64, d['topk_ids']), config=dict(local_start=64))
        for a in arms:
            out = m['br_call'](dd, bank, a)
            e, r = m['br_error'](out, want)
            assert e <= .015 and r <= .075, (size, a, e, r)
            worst = [max(worst[0], e), max(worst[1], r)]
            checks += 1
    print('NEARBY_PASS', worst, flush=True)
    t0 = time.monotonic()
    lines = m['lab'](time.monotonic() + (float(sys.argv[3]) if len(sys.argv) > 3 else 200.), {'w13': w['w13']})
    report = '\n'.join(lines)
    print(report)
    print('LAB_SECONDS', round(time.monotonic() - t0, 1), 'CHARS', len(report))
    (H / f'check{M}{"" if len(sys.argv) < 5 else "_tune"}.json').write_text(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), seeds=seeds, checks=checks,
                                                     bit_equal_to_control=bit, worst_L2_row=worst, arms=arms,
                                                     lab_seconds=round(time.monotonic() - t0, 1), lab_chars=len(report), lines=lines), indent=2) + '\n')


if __name__ == '__main__':
    main()
