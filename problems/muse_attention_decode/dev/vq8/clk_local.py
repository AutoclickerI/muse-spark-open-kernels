"""Phase clocks of vq8 (VQ8_CLK build) on a few scored shapes; output must match the production build.
Run inside tools/img:  python problems/muse_attention_decode/dev/vq8/clk_local.py [--fused]"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'problems/muse_attention_decode'), str(Path(__file__).resolve().parent)]
from local_eval import load_module  # noqa: E402

ref = load_module('reference', ROOT / 'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
import vq8  # noqa: E402

fused = '--fused' in sys.argv  # the main grid merges its own partials (no combine kernel)
for L, b, layer in ((8, 32, 'full'), (8, 1, 'sliding'), (8, 1, 'full'), (1, 32, 'full'), (1, 1, 'sliding')):
    case = dict(batch=b, ctx=80000, heads='tp8', layer=layer, q_len=L, page=1,
                dist='soft' if layer == 'sliding' else 'coherent', **({'spread': 'agentic'} if b > 1 else {}))
    data = ref.generate_input(**case, seed=7)
    k0, v0 = data['k_cache'].clone(), data['v_cache'].clone()
    a = vq8.vq8_attention(data, fused=fused)
    data['k_cache'].copy_(k0), data['v_cache'].copy_(v0)
    rows = []
    c = vq8.vq8_attention(data, clocks=rows, fused=fused)
    torch.cuda.synchronize()
    G = vq8._num_sms(data['qkv'].device)
    print(f'L={L} B={b} {layer} same={bool(torch.equal(a, c))}', vq8.clock_summary(rows[0], G), flush=True)
