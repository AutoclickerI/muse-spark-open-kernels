"""Is vq8's slow consumer start instruction fetch? Clock rows for back-to-back runs, cold vs after an L2 flush."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'problems/muse_attention_decode'), str(Path(__file__).resolve().parent)]
from local_eval import load_module  # noqa: E402

ref = load_module('reference', ROOT / 'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
import vq8  # noqa: E402

flush = torch.empty(1 << 27, dtype=torch.int64, device='cuda')
for L, b, layer in ((8, 1, 'sliding'), (8, 32, 'full')):
    case = dict(batch=b, ctx=80000, heads='tp8', layer=layer, q_len=L, page=1,
                dist='soft' if layer == 'sliding' else 'coherent', **({'spread': 'agentic'} if b > 1 else {}))
    data = ref.generate_input(**case, seed=7)
    G = vq8._num_sms(data['qkv'].device)
    for label in ('warm1', 'warm2', 'flushed', 'warm3'):
        if label == 'flushed':
            flush.sum()
        rows = []
        vq8.vq8_attention(data, clocks=rows)
        torch.cuda.synchronize()
        print(f'L={L} B={b} {layer} {label}', vq8.clock_summary(rows[0], G), flush=True)
