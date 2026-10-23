"""Force every Hopper-only Python branch with graph-safe local stand-ins."""
from pathlib import Path
import ast
import sys
import types
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E

ref = E.load_module('w8_glue_reference', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
source = ast.parse((root / 'dev/xqa_w8_diag_candidate.py').read_text())
source.body.pop()
m = types.ModuleType('w8_glue')
exec(compile(source, str(root / 'dev/xqa_w8_diag_candidate.py'), 'exec'), m.__dict__)
ctx = 2100
as_tuple = False

def standin(**kw):
    q = kw['q']
    cfg = state['config']
    batch, hq, hkv, ps = cfg['batch'], cfg['num_q_heads'], cfg['num_kv_heads'], cfg['page_size']
    ix = torch.arange(ctx + 8, device=q.device)
    rows = kw['page_table'][:, ix // ps].long() * ps + ix % ps
    k = kw['k_cache'].view(-1, hkv, 64)[rows].float().repeat_interleave(8, dim=2)
    v = kw['v_cache'].view(-1, hkv, 64)[rows].float().repeat_interleave(8, dim=2)
    if 'k_descale' in kw:
        k *= kw['k_descale'][:, None, :, None].repeat_interleave(8, dim=2)
    if 'v_descale' in kw:
        v *= kw['v_descale'][:, None, :, None].repeat_interleave(8, dim=2)
    score = torch.einsum('blhd,bkhd->bhlk', q.view(batch, 8, hq, 64).float(), k) * kw['softmax_scale']
    pos = ctx + torch.arange(8, device=q.device)
    mask = ix[None, :] <= pos[:, None]
    if cfg['window']:
        mask &= ix[None, :] >= pos[:, None] - cfg['window'] + 1
    score = score.masked_fill(~mask[None, None], float('-inf'))
    out = torch.einsum('bhlk,bkhd->blhd', score.softmax(-1), v).reshape(batch * 8, hq, 64).to(torch.bfloat16)
    return (out, torch.empty(0, device=q.device)) if as_tuple else out

m.flash_attn_with_kvcache = standin
for layer, page, scale in [('full', 1, 1.), ('sliding', 1, 1.),
                           ('full', 64, .015625), ('sliding', 64, .015625)]:
    state = ref.generate_input(batch=2, ctx=ctx, heads='tp8', layer=layer, q_len=8,
                               page=page, kv_scale=scale, seed=81)
    m._W8_PLAN[(2, bool(state['config']['window']))] = 'fa3'
    for as_tuple in (False, True):
        graph, out = E.capture(m.kernel_fn, state)
        graph.replay()
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(state, ref.collect(state, out))
        print('FA3 stand-in', layer, page, scale, 'tuple', as_tuple, ok, msg, flush=True)
        assert ok, msg
    del graph, out, state

# Stage 5 FP16 needs 104 KiB, beyond sm_120's shared-memory limit. Its Python
# ABI is exercised unchanged, with only its CUDA launch replaced by stage 3.
ext = m._xext()
seen = set()
class ExtensionStandin:
    def xforward(self, *args):
        seen.add((args[-3], args[-2]))
        args = list(args)
        if args[-3] == 5 and args[-2] == 0:
            args[-3] = 3
        return ext.xforward(*args)
m._xext = lambda: ExtensionStandin()
m._w8_input = lambda b, w: ref.generate_input(batch=1 if b == 1 else 2, ctx=128,
    heads='tp8', layer='sliding' if w else 'full', q_len=8, page=1, seed=84)
m._fa3_verify = lambda data: m._xqa_verify(data)
m._w8_diagnose(True)
assert seen == {(st, p8) for st in (3, 4, 5) for p8 in (0, 1)}, seen
assert len(m._W8_PLAN) >= 6
print('PASS: Hopper dispatch, FA3 tensor/tuple, scales, page1/page64, window edges and all tuner variants')
