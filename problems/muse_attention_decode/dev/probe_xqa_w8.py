"""Correctness and repeated fresh-input graph checks for both PV formats."""
import sys
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root.parents[1] / 'tools'), str(root)]
import local_eval as E

ref = E.load_module('probe_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
mod = E.load_module('probe_w8', root / 'dev/xqa_w8_candidate.py')
for p8 in (False, True):
    mod.X_P8 = p8
    for layer, ctx, page, scale, heads, dist in (
        ('full', 6000, 64, 1., 'tp8', 'edges'),
        ('sliding', 4000, 1, 1./64, 'tp8', 'edges'),
        ('full', 20000, 64, 1., 'dp', 'coherent'),
        ('full', 100000, 1, 1., 'tp8', 'coherent'),
        ('full', 1048576, 64, 1., 'tp8', 'soft'),
    ):
        cfg = dict(batch=1 if ctx > 100000 else 2, ctx=ctx, page=page, kv_scale=scale,
                   layer=layer, heads=heads, dist=dist, q_len=8)
        data = ref.generate_input(**cfg, seed=331)
        graph, out = E.capture(mod.kernel_fn, data)
        for seed in (331, 332, 333):
            if seed != 331:
                fresh = ref.generate_input(**cfg, seed=seed)
                E.copy_into(data, fresh)
                del fresh
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
            ok, msg = ref.check_implementation(data, ref.collect(data, out))
            print('PV8', p8, cfg, seed, ok, msg, flush=True)
            assert ok, msg
        del data, graph, out
        torch.cuda.empty_cache()
print('PASS: both PV formats, page1/page64, scales, causal/window edges, 1M context, repeated graph replay')
