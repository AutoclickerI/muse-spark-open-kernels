"""dtype/bytes of every prepared tensor nf_call reads (S7's existing H_H base + recoded NT1 banks) and of the
mixed_parallel bank S7N no longer prepares (s6_bank from S7R's wi_prepare)."""
import importlib.util
import json
import runpy
import sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parents[4]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rows(bank):
    return {k: [str(v.dtype).replace('torch.', ''), list(v.shape), v.numel() * v.element_size()] for k, v in bank.items() if torch.is_tensor(v)}


lab = runpy.run_path(str(R / 'prof/lab/moe_br_m256.py'), init_globals=dict(lab_cold_l2=lambda: None))
n = load('cand_s7n', R / 'problems/muse_decode_layer/cand_r3_best7_s7n.py')
r = load('cand_s7r', R / 'problems/muse_decode_layer/cand_r3_best7_s7r.py')
w = lab['dc_weights'](5353, 'cuda')
wp = dict(w, post_norm_w=w['post_expert_norm'])
base = n.issue_prepare(wp)
native = n._prepare_recoded_bank(w, n.v9_prepare(wp), n._v9_recoded_module())
used = {'s5_hh_base': {k: rows(base)[k] for k in ('w13', 'w13_sf', 'w13_gs', 'w2', 'w2_sf', 'w2_gs', 'pnw')},
        'moe_bank': {k: rows(native)[k] for k in ('w13p', 'w2p')}}
dropped = rows(r.wi_prepare(w))
out = dict(nf_reads=used, nf_reads_bytes=sum(v[2] for b in used.values() for v in b.values()),
           s6_bank_dropped=dropped, s6_bank_bytes=sum(v[2] for v in dropped.values()), moe_bank_grid=native['grid'])
(Path(__file__).parent / 'banks.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out))
