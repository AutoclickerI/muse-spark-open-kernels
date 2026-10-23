"""Compare the frozen K128 and two new families on identical scored inputs."""
from pathlib import Path
import sys
import json
import statistics
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E
ref = E.load_module('pipe_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
mods = {name: E.load_module('pipe_' + name, root / f) for name, f in
        [('old128', 'cand_xqa_k128.py'), ('64', 'dev/xqa_pipeline_candidate.py'),
         ('128', 'dev/xqa_pipeline128_candidate.py')]}
timer = E.Timer()
rows = []
for batch in (16, 32):
    data = ref.generate_input(batch=batch, ctx=80000, heads='tp8', layer='full', q_len=8,
                             spread='agentic', dist='coherent', page=1, seed=43)
    variants = [('old128', False, 3, w) for w in (2, 4, 8)]
    variants += [('64', p8, st, w) for p8 in (False, True) for st in (3, 5, 7) for w in (2, 4, 8)]
    variants += [('128', False, st, w) for st in (3, 4) for w in (2, 4, 8)]
    results = []
    for name, p8, stages, waves in variants:
        m = mods[name]
        m.X_P8, m.X_STAGES, m.X_WAVES = p8, stages, waves
        graph, out = E.capture(m._xqa_verify, data)
        us = statistics.median(timer.replay_us(graph) - timer.fixed_us for _ in range(7))
        results.append((round(us,2),name,p8,stages,waves))
        del graph,out
    results.sort()
    row = dict(batch=batch,variants=results)
    rows.append(row)
    print(json.dumps(row),flush=True)
    (root / 'dev/xqa_pipeline_sweep_local.json').write_text(json.dumps(rows,indent=2))
    del data
    torch.cuda.empty_cache()
