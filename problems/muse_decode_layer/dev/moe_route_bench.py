"""Direct MoE path comparison on decode-layer router fixtures (diagnostic only)."""
import importlib.util
import sys
import statistics
from pathlib import Path

import torch

ROOT = Path('/home/jihoon/competitions/muse_kernel')
P = ROOT / 'problems' / 'muse_decode_layer'
sys.path.insert(0, str(P))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

ref = load('decode_reference_bench', P / 'reference.py')
dec = load('decode_candidate_bench', P / 'cand_r3_v9_autotune.py')
v17 = load('moe_v17_bench', ROOT / 'problems/muse_moe_w4a16/cand_v17.py')
from moe_tune import make_fixture, route_summary, _graph_time


def moe_weights(w):
    return {k: w[k] for k in ('w13','w13_sf','w13_gs','w2','w2_sf','w2_gs')}


def main():
    dev = torch.device('cuda')
    full = ref.generate_weights(ref.WEIGHTS_SEED, device='cuda')
    mw = moe_weights(full)
    mw['post_expert_norm'] = full['post_expert_norm']
    # Decode candidate's proven V7/V9/grouped banks.
    bank9 = dec.v9_prepare({**mw, 'post_norm_w': mw['post_expert_norm']})
    bank7 = dec.v7_prepare({**mw, 'post_norm_w': mw['post_expert_norm']})
    bank9['grouped_mod'] = dec._grouped_module()
    # V17 supplies the Humming/V11 and adaptive paths without running its full portfolio tuner.
    native = v17._prepare_v9({**mw, 'post_norm_w': mw['post_expert_norm']}, 'native')
    adaptive = dict(native); adaptive['mod'] = v17._module('adaptive')
    hstate = v17._prepare_humming({**mw, 'post_norm_w': mw['post_expert_norm']})
    a_nt = {m: v17._adaptive_nt(adaptive, m) for m in (128, 256)}
    flush = torch.empty((1 << 30,), dtype=torch.uint8, device=dev)
    for m in (128, 256):
        for mode in ('uniform', 'matched_uniform', 'real'):
            fixtures = [make_fixture(full, m, 9000 + 7919*i + m, mode=mode) for i in range(3)]
            d0 = fixtures[0]
            data = {k: d0[k] for k in ('hidden_states','topk_ids','topk_weights','config')}
            print('ROUTE', m, mode, route_summary(d0), flush=True)
            paths = {
                'grouped': lambda d: dec._grouped_call(d, bank9),
                'v7': lambda d: dec.experts_v7(d['hidden_states'], d['topk_weights'], d['topk_ids'], {'moe_bank_v7': bank7}),
                'v9': lambda d: dec.experts_v9(d['hidden_states'], d['topk_weights'], d['topk_ids'], {'moe_bank': bank9}),
                'v11_humming': lambda d: v17._kernel_humming(d, hstate),
                'v17_adaptive': lambda d, nt=a_nt[m]: v17._kernel_v9(d, adaptive, 2, nt_override=nt),
            }
            want = paths['v9'](data).float()
            for name, fn in paths.items():
                try:
                    got = fn(data).float()
                    rel = float((got-want).norm()/want.norm().clamp_min(1e-12))
                    t = _graph_time(fn, [{k: x[k] for k in ('hidden_states','topk_ids','topk_weights','config')} for x in fixtures], flush, repeats=4)
                    print('PATH', m, mode, name, 'ms', f'{t:.5f}', 'rel_v9', f'{rel:.6f}', flush=True)
                except Exception as e:
                    print('PATH_FAIL', m, mode, name, type(e).__name__, str(e).splitlines()[0][:200], flush=True)
            del fixtures
            torch.cuda.empty_cache()
    del flush
    torch.cuda.synchronize()

if __name__ == '__main__':
    main()
