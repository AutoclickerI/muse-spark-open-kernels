"""Synthetic projection + boundary/PDL diagnostic for the V9 whole-layer candidate.

This is intentionally a diagnostic, not a submission.  It measures m=128/256 projection
pairs and the O-projection -> residual/norm -> pooler/router boundary with and without the
portable PDL path.  The projection lane's Hopper portfolio winner is represented locally by
its valid dense GEMM stand-in (torch.mm); Hopper-only WGMMA is not inferred from sm_120.
"""
from __future__ import annotations

import importlib.util
import json
import statistics
import time
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "problems/muse_decode_layer"))
REF = ROOT / "problems/muse_decode_layer/reference.py"
V9 = ROOT / "problems/muse_decode_layer/cand_r3_v9.py"
SWEEP = ROOT / "problems/muse_decode_layer/cand_layer_sweep.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph_us(fn, flush, reps=9):
    for _ in range(2):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    vals = []
    for _ in range(reps):
        flush.zero_()
        a.record(); g.replay(); b.record(); b.synchronize()
        vals.append(a.elapsed_time(b) * 1000.0)
    return statistics.median(vals)


def main():
    torch.manual_seed(17)
    ref = load("decode_reference_diag", REF)
    v9 = load("decode_v9_diag", V9)
    sweep = load("decode_sweep_diag", SWEEP)
    dev = torch.device("cuda")
    weights = ref.generate_weights(ref.WEIGHTS_SEED, device="cuda")
    state = v9.prepare({k: v.clone() for k, v in weights.items()})
    flush = torch.empty((128 << 20,), dtype=torch.uint8, device=dev)
    report = {
        "candidate": "cand_r3_v9.py",
        "device": torch.cuda.get_device_name(dev),
        "device_capability": torch.cuda.get_device_capability(dev),
        "projection": {},
        "boundary": {},
        "note": "portfolio H200 WGMMA winners use torch.mm as the portable sm_120 stand-in",
    }

    # These are the projection lane's portfolio matrix pairs.  m=128/256 are deliberately
    # outside the layer's <=64 Triton GEMV region, so the pair exposes the large-M winner.
    projection_weights = {
        "qkv_gate": state["qkv_gate"],
        "o": state["o"],
        "pool_router": state["pool_router"],
        "unpooler": state["unpooler"],
    }
    for m in (128, 256):
        report["projection"][str(m)] = {}
        for name, w in projection_weights.items():
            x = torch.randn((m, w.shape[1]), device=dev, dtype=torch.bfloat16)
            out_dtype = torch.float32 if name == "pool_router" else torch.bfloat16
            # v9 layer path, versus the portable dense stand-in for the portfolio winner.
            layer_us = graph_us(lambda x=x, w=w, d=out_dtype: v9.gemv(x, w, out_dtype=d), flush)
            dense_us = graph_us(lambda x=x, w=w, d=out_dtype: torch.mm(x, w.t()).to(d), flush)
            report["projection"][str(m)][name] = {
                "layer_gemv_us": round(layer_us, 3),
                "portfolio_portable_dense_us": round(dense_us, 3),
                "ratio_dense_over_layer": round(dense_us / layer_us, 4),
            }
            print(f"PROJ m={m} {name}: layer={layer_us:.3f} us portfolio_standin={dense_us:.3f} us", flush=True)

    # Boundary/glue pair: O projection + first residual/norm + pooler/router.  The fused
    # epilogue is taken from the existing structural source and uses the same state ABI.
    for m in (1, 8, 16, 32):
        attn = torch.randn((m, 1024), device=dev, dtype=torch.bfloat16)
        stream = torch.randn((m, 8192), device=dev, dtype=torch.float32)
        row = {}
        for pdl in (False, True):
            v9.PDL = pdl
            sweep.PDL = pdl
            def unfused():
                counts = torch.zeros((15 * m,), device=dev, dtype=torch.int32)
                si = counts[:3 * m].view(3, m)
                sf = counts[3 * m:].view(torch.float32).view(3, m, 4)
                a = v9.gemv(attn, state["o"])
                h, s1 = v9.boundary(a, stream, state["alpha1"], state["beta1"], state["pre_ffn_w"], (sf[0], si[0]))
                pr = v9.gemv(h, state["pool_router"], out_dtype=torch.float32)
                return pr, s1
            def fused():
                counts = torch.zeros((15 * m,), device=dev, dtype=torch.int32)
                si = counts[:3 * m].view(3, m)
                return sweep.o_and_pool(attn, stream, state, si[0])
            u = graph_us(unfused, flush)
            f = graph_us(fused, flush)
            row[f"pdl_{int(pdl)}"] = {"unfused_us": round(u, 3), "fused_us": round(f, 3), "fused_ratio": round(u / f, 4)}
            print(f"BOUNDARY m={m} pdl={int(pdl)}: unfused={u:.3f} us fused={f:.3f} us", flush=True)
        report["boundary"][str(m)] = row
    v9.PDL = True
    sweep.PDL = True
    out = Path(__file__).with_suffix(".json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
