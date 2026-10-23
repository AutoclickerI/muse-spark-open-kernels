"""Compare real local kernels with identical synthetic gathers and cold-L2 graphs."""
import ast
import json
import statistics
import sys
import types
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root.parents[1] / 'tools'), str(root)]
import local_eval as E

src = ast.parse((root / 'dev/xqa_w8_diag_candidate.py').read_text())
src.body.pop()
mod = types.ModuleType('w8timing')
exec(compile(src, 'w8timing.py', 'exec'), mod.__dict__)
old = E.load_module('w8control', root / 'dev/xqa_ptranspose_candidate.py')
timer = E.Timer()
rows = []
for batch, window in ((16, 0), (32, 0), (16, 2048), (32, 2048), (1, 0), (1, 2048)):
    data = mod._w8_input(batch, window)
    results = []
    variants = [('w8', p8, st, waves) for p8 in (False, True)
                for st in ((3, 4, 5) if p8 else (3, 4)) for waves in (1, 2, 4, 8)]
    variants += [('old', False, 5, 2)]
    for family, p8, stages, waves in variants:
        m = mod if family == 'w8' else old
        m.X_P8, m.X_STAGES, m.X_WAVES = p8, stages, waves
        graph, out = E.capture(m._xqa_verify, data)
        us = statistics.median(timer.replay_us(graph) - timer.fixed_us for _ in range(5))
        results.append((round(us, 2), family, p8, stages, waves))
        del graph, out
    results.sort()
    rows.append(dict(batch=batch, window=window, variants=results))
    print(json.dumps(rows[-1]), flush=True)
    (root / 'dev/xqa_w8_sweep_local.json').write_text(json.dumps(rows, indent=2))
    del data
    torch.cuda.empty_cache()
