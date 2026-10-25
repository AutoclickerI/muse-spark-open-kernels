"""S7 H_H (uint8 scale codes, BF16 word formed in-kernel) vs S6 H_H (prepare-widened int16 words) at M=128.

Both candidates' own prepare/plan functions run on identical inputs. W13 activations, W2 outputs and the heavy
experts' norm partials are gathered by (token, slot) pair, so routing's atomic rank order cannot hide a difference,
and must match bit for bit. The final output is compared bit for bit too; the NT1 small-expert norm partials are
float atomics in both, so a rare last-bit difference there is reported separately (S6 vs S6 rerun gives the base rate).
"""
import hashlib
import importlib.util
import json
import runpy
import sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def gathered(plan, ids, start=0):
    v = plan['v13']
    c, off, pos, act, y, ss = v[4], v[5], v[13], v[9], v[15], v[16]
    e = ids.long() - start
    ok = (e >= 0) & (e < 32)
    row = torch.where(ok, off[e.clamp(0, 31)].long() + pos.long(), torch.zeros_like(e))
    heavy = ok & (c[e.clamp(0, 31)] > 16)
    rows = row[ok]
    return act[rows].clone(), y[rows].clone(), ss[row[heavy]].clone()


def run(mod, data, args):
    plan = mod.pm_plan(data, *args, 'H', 'H')
    for f in plan['steps']:
        out = f()
    torch.cuda.synchronize()
    return out.clone(), plan


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    s6 = load('cand_s6', R / 'problems/muse_decode_layer/cand_r3_best7_s6.py')
    s7 = load('cand_s7', R / 'problems/muse_decode_layer/cand_r3_best7_s7.py')
    lab = runpy.run_path(str(R / 'prof/lab/moe_m128_branches.py'))
    stats = dict(fixtures=0, act_bits=0, y_bits=0, ss_heavy_bits=0, out_bit_equal=0, out_rerun_s6_equal=0, worst_L2=0., worst_row=0.)
    for seed in range(seeds):
        w = lab['dc_weights'](5353 + 7919 * seed, 'cuda')
        wp = dict(w, post_norm_w=w['post_expert_norm'])
        base = s7.issue_prepare(wp)
        assert base['w13_sf'].dtype == torch.uint8 and base['w2_sf'].dtype == torch.uint8
        word = s6.issue_word_bank(base)
        native = s7._prepare_recoded_bank(w, s7.v9_prepare(wp), s7._v9_recoded_module())
        real = lab['dc_fixtures'](w, 128)
        b = real[0]
        hot = torch.full_like(b['topk_ids'], -1); hot[:, 0] = 7
        rem = torch.full_like(hot, -1); rem[:40, 0] = 3; rem[:7, 1] = 5; rem[:16, 2] = 9; rem[:17, 3] = 11; rem[:33, 4] = 12; rem[:100, 5] = 20
        cases = real + [dict(b, topk_ids=hot), dict(b, topk_ids=rem)]
        wants = lab['dc_oracles'](cases, w)
        for d, want in zip(cases, wants):
            o6, p6 = run(s6, d, (base, word, native))
            g6 = gathered(p6, d['topk_ids'])
            o6b, _ = run(s6, d, (base, word, native))
            o7, p7 = run(s7, d, (base, native))
            g7 = gathered(p7, d['topk_ids'])
            for name, a, c in zip(('act_bits', 'y_bits', 'ss_heavy_bits'), g6, g7):
                assert torch.equal(a, c), (seed, name, (a != c).sum().item())
                stats[name] += a.numel()
            stats['out_bit_equal'] += bool(torch.equal(o6, o7))
            stats['out_rerun_s6_equal'] += bool(torch.equal(o6, o6b))
            e, r = lab['br_error'](o7, want)
            assert e <= .015 and r <= .075, (seed, e, r)
            stats['worst_L2'] = max(stats['worst_L2'], e)
            stats['worst_row'] = max(stats['worst_row'], r)
            stats['fixtures'] += 1
        print('SEED', seed, stats, flush=True)
    stats.update(seeds=seeds, s6_sha256=hashlib.sha256((R / 'problems/muse_decode_layer/cand_r3_best7_s6.py').read_bytes()).hexdigest(),
                 s7_sha256=hashlib.sha256((R / 'problems/muse_decode_layer/cand_r3_best7_s7.py').read_bytes()).hexdigest())
    (H / 'bitexact.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('BITEXACT_PASS', stats)


if __name__ == '__main__':
    main()
