"""Compare gather-row-base and pipeline copies on the same fresh graphs."""
from pathlib import Path
import statistics
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root.parents[1] / 'tools')]
import local_eval as E
ref = E.load_module('gather_ref', root / 'reference.py')
ref._fa3_scheduler_metadata = lambda *a, **k: None
old = E.load_module('gather_old', root / 'cand_xqa_pipeline.py')
new = E.load_module('gather_new', root / 'cand_xqa_gather.py')
timer = E.Timer()
for batch in (16, 32):
    data = ref.generate_input(batch=batch, ctx=80000, heads='tp8', layer='full', q_len=8,
                              spread='agentic', dist='coherent', page=1, seed=43)
    rows = []
    for name, mod in (('pipeline', old), ('rowbase', new), ('regbase', E.load_module('gather_reg', root / 'cand_xqa_gather2.py'))):
        mod.X_P8 = False; mod.X_STAGES = 3; mod.X_WAVES = 8 if batch == 32 else 4
        graph, out = E.capture(mod._xqa_verify, data)
        vals = [timer.replay_us(graph) - timer.fixed_us for _ in range(9)]
        us = statistics.median(vals)
        ok, msg = ref.check_implementation(data, ref.collect(data, out))
        rows.append((name, round(us, 2), ok, msg))
        del graph, out
    print('GATHER_COMPARE', batch, rows, flush=True)
    del data
    torch.cuda.empty_cache()
