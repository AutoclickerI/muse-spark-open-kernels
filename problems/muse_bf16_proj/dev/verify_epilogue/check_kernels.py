"""Math and graph-replay admission; sm120 timings do not rank H200 variants."""
import json
import sys
from pathlib import Path

import torch
import kernels as V

HERE = Path(__file__).resolve().parent
CONFIGS = [(bm, bn, bk, st, nw, 1) for bm, bn, bk, st, nw in (
    (32, 128, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 64, 4, 4),
    (64, 128, 64, 6, 4), (64, 128, 128, 3, 4), (64, 256, 64, 4, 8),
    (64, 256, 128, 2, 8), (128, 64, 64, 4, 4), (128, 128, 64, 4, 8))]


def main():
    rows = []
    dev = 'cuda'
    gen = torch.Generator(device=dev).manual_seed(18013)
    for m in (128, 256):
        for k in (1024, 4096):
            x = torch.randn(m, k, device=dev, generator=gen).bfloat16()
            w = (torch.randn(8192, k, device=dev, generator=gen) * .02).bfloat16()
            stream = torch.randn(m, 8192, device=dev, generator=gen)
            alpha = torch.randn(8192, device=dev, generator=gen) * .1 + 1
            beta = torch.randn(8192, device=dev, generator=gen) * .1 + 1
            post = torch.randn(8192, device=dev, generator=gen) * .1 + 1 if k == 4096 else None
            pre = torch.randn(8192, device=dev, generator=gen) * .1 + 1
            for cfg in CONFIGS:
                # The local GPU has 99 KiB of shared memory, H200 has 227 KiB.
                # Explicit portable staging stand-in for oversized native cells.
                native_cfg = cfg
                if torch.cuda.get_device_capability() != (9, 0) and cfg[0:3] in ((64, 128, 64), (64, 256, 64), (64, 256, 128), (128, 128, 64)):
                    cfg = (*cfg[:3], min(cfg[3], 2), *cfg[4:])
                def fn():
                    y, part = V.ve_gemm(x, w, stream, alpha, beta, post, cfg)
                    return V.ve_apply(y, part, stream, alpha, beta, pre, post, cfg[1])
                fn()
                torch.cuda.synchronize()
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    out, sout = fn()
                worst = 0.
                for seed in range(3):
                    x.normal_(generator=gen)
                    stream.normal_(generator=gen)
                    gr.replay()
                    br = torch.nn.functional.linear(x, w).float()
                    if post is not None:
                        br *= post
                    br *= (br.square().mean(-1, keepdim=True) + 1.e-8).rsqrt()
                    ref_s = alpha * stream + beta * br
                    ref = (ref_s * (ref_s.square().mean(-1, keepdim=True) + 1.e-5).rsqrt() * pre).bfloat16()
                    err_s = float((sout - ref_s).norm() / ref_s.norm())
                    err_o = float((out.float() - ref.float()).norm() / ref.float().norm())
                    worst = max(worst, err_s, err_o)
                    assert torch.isfinite(out).all() and worst < .004, (m, k, cfg, worst)
                row = dict(m=m, k=k, cfg=native_cfg, executed_cfg=cfg, max_l2=worst, replays=3)
                rows.append(row)
                print(row, flush=True)
    (HERE / 'kernel_validation.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('PASS', len(rows), 'configs; fresh graph fixtures', len(rows) * 3)


if __name__ == '__main__':
    main()
