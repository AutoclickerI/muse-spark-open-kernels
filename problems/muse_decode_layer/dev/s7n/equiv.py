"""S7N's M=128/256 MoE (nf_call on S7's own H_H base + recoded NT1 banks) vs the lab arm LI_swap that H200 701d2867 timed
(prof/lab/moe_nf_tune_m{M}.py on hh_cell's banks). Usage: equiv.py M [seeds]: same inputs, outputs compared bit for bit (the NT1 W2 norm partials
are float atomics, so a rare last-bit difference is counted, not failed), and both gated against the FP32 oracle."""
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


def flush():
    pass


def main():
    M = int(sys.argv[1])
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    cand = load('cand_s7n', R / 'problems/muse_decode_layer/cand_r3_best7_s7n.py')
    lab = runpy.run_path(str(R / f'prof/lab/moe_nf_tune_m{M}.py'), init_globals=dict(lab_cold_l2=flush))
    stats = dict(fixtures=0, bit_equal=0, max_abs_diff=0., worst_L2=0., worst_row=0., lab_worst_L2=0.)
    streams = (torch.cuda.Stream(), torch.cuda.Event(), torch.cuda.Event())
    for seed in range(seeds):
        w = lab['dc_weights'](5353 + 7919 * seed, 'cuda')
        wp = dict(w, post_norm_w=w['post_expert_norm'])
        state = dict(s5_hh_base=cand.issue_prepare(cand.nf_ilv(wp)), moe_bank=cand._prepare_recoded_bank(w, cand.v9_prepare(wp), cand._v9_recoded_module()),
                     nf_streams=streams)
        assert all(state['s5_hh_base'][k].dtype == torch.uint8 for k in ('w13_sf', 'w2_sf'))
        hh = lab['br_prepare'](w)
        real = lab['dc_fixtures'](w, M)
        b = real[0]
        hot = torch.full_like(b['topk_ids'], -1); hot[:, 0] = 7
        rem = torch.full_like(hot, -1)
        for k, (e, n) in enumerate(((3, 40), (5, 7), (9, 16), (11, 17), (12, 33), (20, 9), (21, 8), (22, M - 6))):
            rem[:n, k] = e
        shifted = dict(real[1], topk_ids=torch.where(real[1]['topk_ids'] >= 0, real[1]['topk_ids'] + 96, real[1]['topk_ids']),
                       config=dict(real[1]['config'], local_start=96))
        cases = real + [dict(b, topk_ids=hot), dict(b, topk_ids=rem), shifted]
        wants = lab['dc_oracles'](cases[:-1] + [real[1]], w)
        for d, want in zip(cases, wants):
            oc = cand.nf_call(d, state).clone()
            ol = lab['br_call'](d, hh, 'LI_swap').clone()
            torch.cuda.synchronize()
            e, r = lab['br_error'](oc, want)
            el, rl = lab['br_error'](ol, want)
            assert e <= .015 and r <= .075 and el <= .015 and rl <= .075, (seed, e, r, el, rl)
            diff = float((oc.float() - ol.float()).abs().max())
            assert diff <= 1e-2 * float(ol.float().abs().max()), (seed, diff)
            stats['bit_equal'] += bool(torch.equal(oc, ol))
            stats['max_abs_diff'] = max(stats['max_abs_diff'], diff)
            stats['fixtures'] += 1
            stats['worst_L2'] = max(stats['worst_L2'], e)
            stats['worst_row'] = max(stats['worst_row'], r)
            stats['lab_worst_L2'] = max(stats['lab_worst_L2'], el)
        print('SEED', seed, stats, flush=True)
    stats.update(M=M, seeds=seeds, s7n_sha256=hashlib.sha256((R / 'problems/muse_decode_layer/cand_r3_best7_s7n.py').read_bytes()).hexdigest())
    (H / f'equiv{M}.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('EQUIV_PASS', stats)


if __name__ == '__main__':
    main()
