"""Local graph/reference test for the Muse prefill FP8 attention adapter."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _load():
    path = Path(__file__).parents[1] / "sglang/python/sglang/srt/layers/muse_prefill_attention.py"
    spec = importlib.util.spec_from_file_location("muse_prefill_attention", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    d = torch.device("cuda")
    B, L, HQ, HK, D, CTX = 1, 2048, 16, 2, 64, 256
    q = torch.randn(B * L, HQ, D, device=d, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    k = torch.randn(CTX + L, 1, HK, D, device=d, dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    v = torch.randn(CTX + L, 1, HK, D, device=d, dtype=torch.bfloat16).to(torch.float8_e4m3fn).contiguous()
    pt = torch.arange(CTX + L, device=d, dtype=torch.int32).view(1, -1)
    cs = torch.tensor([CTX + L], device=d, dtype=torch.int32)
    cu = torch.tensor([0, L], device=d, dtype=torch.int32)
    mod = _load()
    for window in (-1, 2047):
        out = mod.prefill_attention(
            q, k, v, pt, cs, cu, torch.ones((), device=d), torch.ones((), device=d),
            softmax_scale=0.125, window_left=window, page_size=1,
        )
        torch.cuda.synchronize()
        t = L - 1
        ids = torch.arange(0, CTX + t + 1, device=d) if window < 0 else torch.arange(max(0, CTX + t - window), CTX + t + 1, device=d)
        kk, vv = k[ids].float().view(-1, HK, D), v[ids].float().view(-1, HK, D)
        ref = []
        for hk in range(HK):
            logits = torch.einsum("gd,sd->gs", q[t, hk * 8 : (hk + 1) * 8].float(), kk[:, hk]) * 0.125
            ref.append(torch.einsum("gs,sd->gd", logits.softmax(-1), vv[:, hk]))
        err = float((out[t].float() - torch.cat(ref).float()).norm() / torch.cat(ref).float().norm())
        assert err < 0.04 and torch.isfinite(out).all()
        replay = torch.empty_like(out)
        for _ in range(2):
            mod.prefill_attention(q, k, v, pt, cs, cu, torch.ones((), device=d), torch.ones((), device=d), softmax_scale=0.125, window_left=window, page_size=1, out=replay)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mod.prefill_attention(q, k, v, pt, cs, cu, torch.ones((), device=d), torch.ones((), device=d), softmax_scale=0.125, window_left=window, page_size=1, out=replay)
        graph.replay(); torch.cuda.synchronize()
        assert torch.equal(replay.view(torch.uint8), out.view(torch.uint8))
        print(f"MUSE_PREFILL graph window={window}: last-row rel_l2={err:.3e} PASS")


if __name__ == "__main__":
    main()
