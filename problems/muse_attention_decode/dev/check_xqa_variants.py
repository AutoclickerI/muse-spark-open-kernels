"""Check the gather-ring/split variants that fit the local GPU against the spec."""
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E
ref = E.load_module('w8_variant_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
mod = E.load_module('w8_variant_mod', root / 'cand_xqa_k128.py')
for cfg in (
    dict(batch=3, ctx=17, page=64, layer='full', dist='edges'),
    dict(batch=3, ctx=5000, page=1, layer='sliding', dist='edges', spread='mixed'),
    dict(batch=2, ctx=20000, page=1, layer='full', dist='coherent', heads='dp'),
):
    cfg = dict(heads='tp8', q_len=8) | cfg
    data = ref.generate_input(**cfg, seed=996)
    for p8 in (False, True):
        for st in ((3, 4, 5) if p8 else (3, 4)):
            for waves in (1, 8):
                mod._w8_set((p8, st, waves))
                graph, out = E.capture(mod._xqa_verify, data)
                for _ in range(5):
                    graph.replay()
                torch.cuda.synchronize()
                ok, msg = ref.check_implementation(data, ref.collect(data, out))
                print(cfg, p8, st, waves, ok, msg, flush=True)
                assert ok, msg
                del graph, out
    del data
    torch.cuda.empty_cache()
print('PASS: all local gather stages and split extremes, short contexts, empty splits, window edges and DP heads')
