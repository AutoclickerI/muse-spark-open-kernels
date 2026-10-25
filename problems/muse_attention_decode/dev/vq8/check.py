"""Local gate for vq8: TASK.md's q_len-8 tests, the scored verify shapes, graph capture + fresh inputs.
Run inside tools/img:  python problems/muse_attention_decode/dev/vq8/check.py [--quick] [--fused]"""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'problems/muse_attention_decode'), str(Path(__file__).resolve().parent)]
from local_eval import load_module, copy_into  # noqa: E402

ref = load_module('reference', ROOT / 'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
import vq8  # noqa: E402

quick = '--quick' in sys.argv
CASES = [
    dict(batch=5, ctx=5000, heads='tp8', layer='sliding', q_len=8, spread='mixed', page=1),
    dict(batch=2, ctx=100000, heads='tp8', layer='full', q_len=8, page=1),
    dict(batch=4, ctx=6000, heads='tp8', layer='full', q_len=8, dist='edges'),
    dict(batch=1, ctx=1048576, heads='tp8', layer='full', q_len=8),
    dict(batch=3, ctx=4000, heads='tp8', layer='sliding', q_len=8, page=1, kv_scale=0.015625),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', q_len=8, dist='soft', page=1),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='soft'),
    dict(batch=2, ctx=20000, heads='dp', layer='full', q_len=8, dist='coherent'),
    dict(batch=3, ctx=10, heads='tp8', layer='sliding', q_len=8),
    dict(batch=2, ctx=70, heads='tp8', layer='full', q_len=8, page=1),
    dict(batch=1, ctx=10, heads='tp8', layer='sliding'),
    dict(batch=3, ctx=2047, heads='tp8', layer='sliding'),
    dict(batch=4, ctx=3000, heads='tp8', layer='full'),
    dict(batch=8, ctx=8192, heads='dp', layer='full', spread='mixed'),
    dict(batch=4, ctx=6000, heads='tp8', layer='sliding', dist='edges', page=1),
    dict(batch=2, ctx=9000, heads='tp8', layer='full', kv_scale=0.015625),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='soft'),
    dict(batch=4, ctx=80000, heads='tp8', layer='full', spread='mixed', dist='coherent', page=1),
]
SCORED = [dict(batch=b, ctx=80000, heads='tp8', layer=layer, q_len=L, page=1, dist='soft' if layer == 'sliding' else 'coherent',
               **({'spread': 'agentic'} if b > 1 else {})) for L in (8, 1) for b in (1, 16, 32) for layer in ('sliding', 'full')]
if quick:
    CASES = CASES[:1] + CASES[4:6] + CASES[8:]
    SCORED = SCORED[:2] + SCORED[6:8]

LOCAL = torch.cuda.get_device_capability()[0] != 9
settings = [dict(min_tiles=2, stages=2 if LOCAL else 8), dict(min_tiles=1, stages=2 if LOCAL else 4, grid=7), dict(min_tiles=3, stages=2 if LOCAL else 6, grid=40)]
if '--fused' in sys.argv:  # the main grid merges its own partials (no combine kernel)
    settings = [dict(st, fused=True) for st in settings]
fails = 0
for i, case in enumerate(CASES + SCORED):
    for st in (settings if i < len(CASES) else settings[:1]):
        t0 = time.time()
        data = ref.generate_input(**case, seed=1000 + i)
        g = torch.cuda.CUDAGraph()
        out = vq8.vq8_attention(data, **st)  # warm
        torch.cuda.synchronize()
        fresh0 = ref.generate_input(**case, seed=1000 + i)
        copy_into(data, fresh0)
        with torch.cuda.graph(g):
            out = vq8.vq8_attention(data, **st)
        fresh = ref.generate_input(**case, seed=2000 + i)
        copy_into(data, fresh)
        g.replay()
        torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        fails += not ok
        print(('PASS' if ok else 'FAIL'), case, st, msg, f'{time.time() - t0:.1f}s', flush=True)
print('FAILS', fails)
