"""S7R's M=256 MoE (rf_call) vs S7's mixed_parallel (wi_call policy 6, grid 2112) on the same bank and inputs:
outputs must match bit for bit (real routes, one hot expert, remainders of every class, nonzero local_start)."""
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


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    cand = load('cand_s7r', R / 'problems/muse_decode_layer/cand_r3_best7_s7r.py')
    lab = runpy.run_path(str(R / 'prof/lab/moe_br_m256.py'))
    stats = dict(fixtures=0, bit_equal=0, worst_L2=0., worst_row=0.)
    for seed in range(seeds):
        w = lab['dc_weights'](5353 + 7919 * seed, 'cuda')
        bank = cand.wi_prepare(w)
        real = lab['dc_fixtures'](w, 256)
        b = real[0]
        hot = torch.full_like(b['topk_ids'], -1); hot[:, 0] = 7
        rem = torch.full_like(hot, -1)
        for k, (e, n) in enumerate(((3, 40), (5, 7), (9, 16), (11, 17), (12, 33), (20, 9), (21, 8), (22, 250))):
            rem[:n, k] = e
        shifted = dict(real[1], topk_ids=torch.where(real[1]['topk_ids'] >= 0, real[1]['topk_ids'] + 96, real[1]['topk_ids']),
                       config=dict(real[1]['config'], local_start=96))
        cases = real + [dict(b, topk_ids=hot), dict(b, topk_ids=rem), shifted]
        wants = lab['dc_oracles'](cases[:-1] + [real[1]], w)
        for d, want in zip(cases, wants):
            o6 = cand.wi_call(d, bank, 6, 2112).clone()
            orf = cand.rf_call(d, bank).clone()
            torch.cuda.synchronize()
            assert torch.equal(o6, orf), (seed, (o6 != orf).sum().item())
            e, r = lab['br_error'](orf, want)
            assert e <= .015 and r <= .075, (seed, e, r)
            stats['bit_equal'] += 1
            stats['fixtures'] += 1
            stats['worst_L2'] = max(stats['worst_L2'], e)
            stats['worst_row'] = max(stats['worst_row'], r)
        print('SEED', seed, stats, flush=True)
    stats.update(seeds=seeds, s7r_sha256=hashlib.sha256((R / 'problems/muse_decode_layer/cand_r3_best7_s7r.py').read_bytes()).hexdigest())
    (H / 'bitexact.json').write_text(json.dumps(stats, indent=2) + '\n')
    print('BITEXACT_PASS', stats)


if __name__ == '__main__':
    main()
