"""Force every family, ring size, split extreme and Hopper dispatch locally."""
from pathlib import Path
import json
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E

path = root / 'cand_xqa_next.py'
ref = E.load_module('next_verify_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
m = E.load_module('next_verify_mod', path)
variants = [('group', s, w) for s in (5, 7) for w in (1, 8)]
variants += [('warp', s, w) for s in (3, 4) for w in (1, 8)]
rows = []
for kwargs in (
    dict(batch=3, ctx=17, heads='tp8', layer='full', page=64, dist='edges'),
    dict(batch=3, ctx=5000, heads='tp8', layer='sliding', page=1, dist='edges', spread='mixed'),
    dict(batch=2, ctx=20000, heads='dp', layer='full', page=64, dist='coherent', kv_scale=.015625),
    dict(batch=1, ctx=1048576, heads='tp8', layer='full', page=64, dist='coherent'),
):
    data = ref.generate_input(**kwargs, q_len=8, seed=9919)
    for variant in variants:
        m._NEXT_PLAN[kwargs['batch'], kwargs['layer'] == 'sliding'] = variant
        graph, out = E.capture(m.kernel_fn, data)
        for _ in range(5): graph.replay()
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        assert ok, (kwargs, variant, msg)
        rows.append(dict(case=kwargs, variant=variant, ok=ok, msg=msg))
        print('NEXT_VARIANT', kwargs, variant, ok, msg, flush=True)
        del graph, out
    del data
    torch.cuda.empty_cache()
(root / 'dev/xqa_next_variants.json').write_text(json.dumps(rows, indent=2))

# Force the Hopper FA3 branch with tensor and tuple returns, both page sizes,
# and scaled KV. No host reads appear inside the callable stand-in.
state = None
as_tuple = False
ctx = 2100
def standin(**kw):
    cfg = state['config']; batch = cfg['batch']; hq = cfg['num_q_heads']
    hkv = cfg['num_kv_heads']; ps = cfg['page_size']; q = kw['q']
    ix = torch.arange(ctx + 8, device=q.device)
    physical = kw['page_table'][:, ix // ps].long() * ps + ix % ps
    k = kw['k_cache'].view(-1, hkv, 64)[physical].float().repeat_interleave(8, dim=2)
    v = kw['v_cache'].view(-1, hkv, 64)[physical].float().repeat_interleave(8, dim=2)
    if 'k_descale' in kw: k *= kw['k_descale'][:, None, :, None].repeat_interleave(8, dim=2)
    if 'v_descale' in kw: v *= kw['v_descale'][:, None, :, None].repeat_interleave(8, dim=2)
    scores = torch.einsum('blhd,bkhd->bhlk', q.view(batch, 8, hq, 64).float(), k) * kw['softmax_scale']
    pos = ctx + torch.arange(8, device=q.device)
    mask = ix[None, :] <= pos[:, None]
    if cfg['window']: mask &= ix[None, :] >= pos[:, None] - cfg['window'] + 1
    scores = scores.masked_fill(~mask[None, None], float('-inf'))
    out = torch.einsum('bhlk,bkhd->blhd', scores.softmax(-1), v).reshape(batch * 8, hq, 64).to(torch.bfloat16)
    return (out, torch.empty(0, device=q.device)) if as_tuple else out
m.flash_attn_with_kvcache = standin
for layer, page, scale in [('full', 1, 1.), ('sliding', 1, 1.),
                           ('full', 64, .015625), ('sliding', 64, .015625)]:
    state = ref.generate_input(batch=2, ctx=ctx, heads='tp8', layer=layer,
                               q_len=8, page=page, kv_scale=scale, seed=9921)
    m._NEXT_PLAN[2, layer == 'sliding'] = 'fa3'
    for as_tuple in (False, True):
        graph, out = E.capture(m.kernel_fn, state)
        for _ in range(3): graph.replay()
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(state, ref.collect(state, out))
        assert ok, msg
        print('NEXT_FA3_GLUE', layer, page, scale, as_tuple, ok, flush=True)
        del graph, out

# Exercise prepare's actual tuning/graph/paired-choice control flow on small
# synthetic inputs, preserving all family and stage choices. The FA3 callable
# is replaced by the already validated portable kernel at its data ABI.
m._next_input = lambda b, w: ref.generate_input(batch=1 if b == 1 else 2,
    ctx=128, heads='tp8', layer='sliding' if w else 'full', q_len=8, page=1, seed=9931)
seen = set()
original_set = m._next_set
def record(v):
    seen.add(v)
    return original_set(v)
m._next_set = record
m._fa3_verify = lambda data: m._xqa_verify(data)
m._next_diagnose()
assert len(m._NEXT_PLAN) >= 6
assert seen == ({('group', s, w) for s in (5, 7) for w in (1, 2, 4, 8)} |
                {('warp', s, w) for s in (3, 4) for w in (1, 2, 4, 8)})
print('PASS: both families, ring sizes, split extremes, 1M context, FA3 glue and all tuner choices')
