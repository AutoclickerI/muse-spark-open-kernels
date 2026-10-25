"""Local check of a patched decode-layer file's vq8 cell: the layer's own prepare gate (_vq8_prepare) and the
attention TASK gate incl. new K/V at the routed scored shapes, on FP32 split partials (S=7, strided) and BF16
(S=1), CUDA-graph captured and replayed on fresh inputs.  usage: check_cell.py <patched.py>"""
import importlib.util
import sys
from pathlib import Path
import torch
ROOT = Path('/home/jihoon/competitions/muse_kernel')
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'problems/muse_attention_decode')]
from local_eval import load_module, copy_into
ref = load_module('reference', ROOT / 'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
spec = importlib.util.spec_from_file_location('cell', sys.argv[1])
cell = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cell)
dev = torch.device('cuda', 0)
rows = cell._vq8_prepare(dev)
fails = 0 if rows == cell._VQ8_ROWS else 1
print('prepare gate rows', rows)
for (B, L, sliding), setting in cell._VQ8_ROWS.items():
    layer = 'sliding' if sliding else 'full'
    for S in (7, 1):
        case = dict(batch=B, ctx=80000, heads='tp8', layer=layer, q_len=L, page=1,
                    dist='soft' if sliding else 'coherent', **({'spread': 'agentic'} if B > 1 else {}))
        data = ref.generate_input(**case, seed=70 + B + S)
        T, W = B * L, data['qkv'].shape[-1]
        def make(d, buf):
            if S == 7:
                P = buf[:, :, :W]
                P.copy_(torch.randn_like(P) * 0.01)
                P[-1] += d['qkv'].view(T, W).float() - P.sum(0)
                d['qkv'] = P.sum(0).to(torch.bfloat16).view(B, L, W)
                return (P, S)
            buf.copy_(d['qkv'].view(1, T, W))
            return (buf, 1)
        buf = torch.empty((7, T, W + 64), device=dev) if S == 7 else torch.empty((1, T, W), device=dev, dtype=torch.bfloat16)
        arg = make(data, buf)
        cell._vq8_attention(arg, data, setting); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = cell._vq8_attention(arg, data, setting)
        fresh = ref.generate_input(**case, seed=170 + B + S)
        arg2 = make(fresh, buf)
        copy_into(data, fresh)
        data['qkv'] = fresh['qkv']
        g.replay(); torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out.view(B, L, -1)))
        fails += not ok
        print('PASS' if ok else 'FAIL', (B, L, layer), 'setting', setting, 'splits', S, msg[:110], flush=True)
print('FAILS', fails)
