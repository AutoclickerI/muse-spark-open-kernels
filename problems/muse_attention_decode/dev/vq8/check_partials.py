"""Decode-layer ABI gate for attention_vq8: FP32 projection partials (strided), BF16 single split, graphs.
[--fused]: the fused-merge build path."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'problems/muse_attention_decode'), str(Path(__file__).resolve().parent)]
from local_eval import load_module, copy_into
ref = load_module('reference', ROOT / 'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata = lambda *a, **kw: None
import vq8
fails = 0
FUSED = '--fused' in sys.argv  # the main grid merges its own partials (no combine kernel)
for B, layer in ((1, 'sliding'), (16, 'full'), (32, 'sliding'), (3, 'full')):
    for S in (0, 1, 3):
        case = dict(batch=B, ctx=80000 if B != 3 else 3000, heads='tp8', layer=layer, q_len=8, page=1,
                    dist='coherent' if layer == 'full' else 'soft', **({'spread': 'agentic'} if B > 1 else {}))
        data = ref.generate_input(**case, seed=40 + B + S)
        T, W = B * 8, data['qkv'].shape[-1]
        if S:
            buf = torch.randn((S, T, W + 64), device='cuda') * 0.01  # strided: padded rows
            P = buf[:, :, :W]
            P[-1] += data['qkv'].view(T, W).float() - P.sum(0)
            data['qkv'].copy_(P.sum(0).to(torch.bfloat16).view_as(data['qkv']))
            arg = (P, S)
        else:
            arg = (data['qkv'].view(1, T, W), 1)
        g = torch.cuda.CUDAGraph()
        vq8.attention_vq8(arg, data, fused=FUSED); torch.cuda.synchronize()
        with torch.cuda.graph(g):
            out = vq8.attention_vq8(arg, data, fused=FUSED)
        fresh = ref.generate_input(**case, seed=90 + B + S)
        if S:
            P.copy_(torch.randn_like(P) * 0.01); P[-1] += fresh['qkv'].view(T, W).float() - P.sum(0)
            fresh['qkv'] = P.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
        copy_into(data, fresh)
        g.replay(); torch.cuda.synchronize()
        ok, msg = ref.check_implementation(data, ref.collect(data, out.view(B, 8, -1)))
        fails += not ok
        print('PASS' if ok else 'FAIL', B, layer, 'splits', S, msg[:90], flush=True)
        # No-gate mode (the gate split): RMSNorm only; times sigmoid(gate) it must match the gated output.
        if S == 1:
            HQ = data['config']['num_q_heads']
            gate = data['qkv'].view(T, W)[:, W - HQ * 64:].float()
            ng = vq8.attention_vq8(arg, data, gate=False, fused=FUSED).float()
            torch.cuda.synchronize()
            err = float((ng * torch.sigmoid(gate) - out.float()).norm() / out.float().norm())
            fails += not err < 1e-2
            print('PASS' if err < 1e-2 else 'FAIL', B, layer, 'no-gate rel err %.2e' % err, flush=True)
print('FAILS', fails)
