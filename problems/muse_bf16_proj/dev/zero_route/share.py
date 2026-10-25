"""Share of tokens with no local route (all 8 experts off this GPU) on the decode-layer shapes.

    tools/img python problems/muse_bf16_proj/dev/zero_route/share.py

Uses reference.py's inputs and routing (local_start = the busiest 32-expert range, as case_fields), on the
default router and on a sweep of router seeds (the judge picks one per run to hit hidden load targets).
"""
import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('reference', REPO / 'problems/muse_decode_layer/reference.py')
ref = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(REPO / 'problems/muse_decode_layer'))
spec.loader.exec_module(ref)
benches = json.loads((REPO / 'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks']
dev = 'cuda'
weights = ref.generate_weights(1, dev)
hs = {}
for b in benches:
    if b['layer'] != 'sliding':
        continue
    case = {k: v for k, v in b.items() if k not in ('load', 'weight', 'repeats')}
    data = ref.generate_input(**{**case, 'local_start': 0})
    hs[b['load']] = ref.ref_front(data, weights)['h']


def stats(h, router):
    ids = ref.route(h, router)[1].topk(8, dim=-1).indices
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    start = 32 * int(pairs.argmax())
    local = ((ids - start) % 256) < 32
    per_tok = local.sum(1)
    return int(local.sum()), float((per_tok == 0).float().mean()), h.shape[0]


rows = {}
default = {'router': weights['router'], 'router_bias': weights['router_bias']}
for seed in [None] + list(range(1, 41)):
    r = default if seed is None else ref._router(torch.Generator(device=dev).manual_seed(seed * 0x9E3779B97F4A7C15 % (2**63 - 1)), dev, weights['pre_ffn_norm'])
    for load, h in hs.items():
        routes, zero, t = stats(h, r)
        rows.setdefault(load, []).append((routes, zero, t))
for load, v in rows.items():
    t = v[0][2]
    zs = sorted(z for _, z, _ in v)
    rs = sorted(r for r, _, _ in v)
    print(f'{load}: T={t} default routes={v[0][0]} ({v[0][0] / (8 * t):.3f}) zero-local={v[0][1]:.3f} | '
          f'40 seeds: routes {rs[0]}-{rs[-1]} median {rs[len(rs) // 2]}, zero-local {zs[0]:.3f}-{zs[-1]:.3f} median {zs[len(zs) // 2]:.3f}')
