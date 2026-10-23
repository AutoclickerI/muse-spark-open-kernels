"""Local sm120 comparison of readers of the single canonical V6 expert bank.

The frozen MoE lane sources are under canonical_baselines/. Their grouped source is
adapted only to add the separate gate/up FP32 scale pointers required by serving.
Run in the judge image, e.g. tools/img python <this file> --tokens 753 3012 16384.
No local timing in this file establishes a Hopper performance result.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import nvfp4

H = 4096
E = 32
_MODULES = {}
_FLUSH = None


def frozen_python(kind="rs"):
    path = HERE / "canonical_baselines" / f"cand_{kind}.py"
    name = "canonical_frozen_" + kind
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def separate_global_scales(source):
    replacements = (
        ("const uint8_t* weights,const float* gs,const __nv_bfloat16* x",
         "const uint8_t* weights,const float* gs,const float* gs_up,const __nv_bfloat16* x"),
        ("const float sc=gs[e]*256.f;", "const float sc=gs[e]*256.f,scu=gs_up[e]*256.f;"),
        ("float u=__bfloat162float(__float2bfloat16_rn(acc[4*j+2+q]*sc));",
         "float u=__bfloat162float(__float2bfloat16_rn(acc[4*j+2+q]*scu));"),
        ("float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*sc));",
         "float u=__bfloat162float(__float2bfloat16_rn(acc[rb][4*j+2+q]*scu));"),
        ("torch::Tensor w,torch::Tensor gs,torch::Tensor x", "torch::Tensor w,torch::Tensor gs,torch::Tensor gs_up,torch::Tensor x"),
        ("gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr()",
         "gs.data_ptr<float>(),gs_up.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr()"),
    )
    for old, new in replacements:
        # Each source has exactly one of the two epilogue spellings.
        if "float u=" not in old:
            assert source.count(old) == 1, (old, source.count(old))
        source = source.replace(old, new)
    assert "float u=" in source and "*scu));" in source
    return source


def baseline_module(kind):
    if kind not in _MODULES:
        from torch.utils.cpp_extension import load_inline
        source = separate_global_scales(frozen_python(kind).GROUPED_SRC)
        cap = torch.cuda.get_device_capability()
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a" if cap == (9, 0) else f"{cap[0]}.{cap[1]}"
        try:
            module = load_inline(
                name="canonical_comparison_" + hashlib.sha256(source.encode()).hexdigest()[:14],
                cpp_sources=(
                    "void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n"
                    "void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);"
                ),
                cuda_sources=source, functions=["route", "gemm"],
                extra_cuda_cflags=["-O3", "-std=c++20", "-lineinfo"], verbose=False,
            )
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
        _MODULES[kind] = module
    return _MODULES[kind]


@triton.jit
def finalize(Y, POS, PROBS, NORM, OUT, D: tl.constexpr):
    t = tl.program_id(0)
    c = tl.arange(0, D)
    norm = tl.load(NORM + c).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for k in range(8):
        pos = tl.load(POS + t * 8 + k)
        y = tl.load(Y + pos.to(tl.int64) * D + c, pos >= 0, 0).to(tl.float32) * norm
        y *= tl.rsqrt(tl.sum(y * y, 0) / D + 1e-8)
        acc += tl.load(PROBS + t * 8 + k) * y
    tl.store(OUT + t * D + c, acc.to(tl.bfloat16))


def make_weights(seed=7151, device="cuda", unequal=True):
    """A single 864 MiB canonical bank, with independent gate/up global scales."""
    gen = torch.Generator(device=device).manual_seed(seed)
    packer = frozen_python("rs")
    rowmap13 = torch.cat((torch.arange(4096, device=device).view(512, 8),
                         torch.arange(4096, 8192, device=device).view(512, 8)), 1)
    rowmap2 = torch.arange(4096, device=device).view(256, 16)
    w13p = torch.empty((E, 512, 36864), dtype=torch.uint8, device=device)
    w2p = torch.empty((E, 256, 36864), dtype=torch.uint8, device=device)
    # Keep raw weights only for the independent FP32 reference, not in reader state.
    raw13, rawsf13, raw2, rawsf2 = [], [], [], []
    for e in range(E):
        w13, sf13, base = nvfp4.random_weights(8192, H, generator=gen, device=device)
        w2, sf2, _ = nvfp4.random_weights(H, H, generator=gen, device=device)
        w13p[e] = packer._repack(w13, sf13, rowmap13)
        w2p[e] = packer._repack(w2, sf2, rowmap2)
        raw13.append(w13); rawsf13.append(sf13)
        raw2.append(w2); rawsf2.append(sf2)
    gs13_gate = base * torch.exp(torch.randn((E,), generator=gen, device=device) * .3)
    gs13_up = gs13_gate * (1.375 if unequal else 1.0)
    gs2 = base * torch.exp(torch.randn((E,), generator=gen, device=device) * .3)
    norm = (1 + .1 * torch.randn((H,), generator=gen, device=device)).to(torch.bfloat16)
    state = dict(w13p=w13p, w2p=w2p, gs13_gate=gs13_gate, gs13_up=gs13_up,
                 gs2=gs2, post_expert=norm,
                 sms=torch.cuda.get_device_properties(device).multi_processor_count)
    raw = dict(w13=raw13, sf13=rawsf13, w2=raw2, sf2=rawsf2)
    return state, raw


def make_fixture(tokens, seed=291, device="cuda", distribution="uniform"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((tokens, H), generator=gen, device=device).to(torch.bfloat16)
    # Uniform top8 over 256 experts: near one local pair/token. Balanced and hot
    # modes are useful for padding/occupancy diagnostics, without changing math.
    if distribution == "uniform":
        scores = torch.rand((tokens, 256), generator=gen, device=device)
        ids = scores.topk(8, dim=-1).indices.to(torch.int32)
    elif distribution == "balanced":
        ids = torch.full((tokens, 8), 255, dtype=torch.int32, device=device)
        ids[:, 0] = torch.arange(tokens, device=device) % E
    elif distribution == "hot":
        ids = torch.full((tokens, 8), 255, dtype=torch.int32, device=device)
        ids[:, 0] = 0
    else:
        raise ValueError(distribution)
    probs = torch.rand((tokens, 8), generator=gen, device=device)
    probs /= probs.sum(-1, keepdim=True)
    return dict(x=x, ids=ids, probs=probs)


def workspace(fixture):
    x = fixture["x"]
    count = x.shape[0] * 8
    d = x.device
    return dict(counts=torch.empty(32, dtype=torch.int32, device=d),
                offsets=torch.empty(33, dtype=torch.int32, device=d),
                cursors=torch.empty(32, dtype=torch.int32, device=d),
                pairs=torch.empty(count, dtype=torch.int32, device=d),
                positions=torch.empty(count, dtype=torch.int32, device=d),
                tasks=torch.empty(5 * 33, dtype=torch.int32, device=d),
                inter=torch.empty((count, H), dtype=torch.bfloat16, device=d),
                y=torch.empty((count, H), dtype=torch.bfloat16, device=d),
                out=torch.empty_like(x))


def run_baseline(kind, fixture, state, work, fixed_n=0, stages=2, stage="all"):
    mod = baseline_module(kind)
    if stage in ("all", "route"):
        mod.route(fixture["ids"], work["counts"], work["offsets"], work["cursors"],
                  work["pairs"], work["positions"], work["tasks"], 0, fixed_n)
    if stage in ("all", "gate"):
        mod.gemm(state["w13p"], state["gs13_gate"], state["gs13_up"], fixture["x"],
                 work["inter"], work["pairs"], work["offsets"], work["tasks"],
                 1, fixed_n, state["sms"] * 4, stages)
    if stage in ("all", "down"):
        mod.gemm(state["w2p"], state["gs2"], state["gs2"], work["inter"],
                 work["y"], work["pairs"], work["offsets"], work["tasks"],
                 0, fixed_n, state["sms"] * 4, stages)
    if stage in ("all", "finalize"):
        finalize[(fixture["x"].shape[0],)](work["y"], work["positions"], fixture["probs"],
                                            state["post_expert"], work["out"], D=H, num_warps=8)
    return work["out"]


@triton.jit
def slot_metadata(IDS, POS, OFFSETS, SLOT_E, SLOT_POS, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    e = tl.load(IDS + i, i < SIZE, 255)
    valid = e < 32
    off = tl.load(OFFSETS + e, valid, 0)
    pos = tl.load(POS + i, i < SIZE, -1)
    tl.store(SLOT_E + i, tl.where(valid, e, -1), i < SIZE)
    tl.store(SLOT_POS + i, tl.where(valid, pos - off, 0), i < SIZE)


def custom_host():
    import moe_canonical_host
    return moe_canonical_host


def custom_workspace(fixture):
    work = workspace(fixture)
    t = fixture["x"].shape[0]
    dev = fixture["x"].device
    tiles = triton.cdiv(t * 8 + 32 * 127, 128)
    work.update(tstart=torch.empty(33, dtype=torch.int32, device=dev),
                slot_e=torch.empty((t, 8), dtype=torch.int32, device=dev),
                slot_pos=torch.empty((t, 8), dtype=torch.int32, device=dev),
                rows=torch.empty((t, 8), dtype=torch.int32, device=dev),
                xt=torch.empty((tiles * 128 * H,), dtype=torch.bfloat16, device=dev),
                inter=torch.empty((tiles * 128 * H,), dtype=torch.bfloat16, device=dev),
                y=torch.empty((tiles * 128 * H,), dtype=torch.bfloat16, device=dev))
    return work


def run_custom(fixture, state, work, stage="all"):
    host = custom_host()
    t = fixture["x"].shape[0]
    if stage in ("all", "route"):
        baseline_module("rs").route(fixture["ids"], work["counts"], work["offsets"], work["cursors"],
                                     work["pairs"], work["positions"], work["tasks"], 0, 128)
        slot_metadata[(triton.cdiv(t * 8, 256),)](fixture["ids"], work["positions"], work["offsets"],
                                                 work["slot_e"], work["slot_pos"], SIZE=t * 8, BLOCK=256)
    if stage in ("all", "gather"):
        host.tile_offsets_kernel[(1,)](work["counts"], work["tstart"], BNC=128)
        host.gather_tiled_kernel[(t,)](fixture["x"], work["slot_e"], work["slot_pos"], work["tstart"],
                                       work["xt"], work["rows"], D=H, BNC=128, num_warps=4)
    if stage in ("all", "gate"):
        state["tiled_mod"].moe_gemm(state["w13p"], state["gs13_gate"], state["gs13_up"],
                                    work["xt"], work["inter"], work["tstart"], 32, 64, 64, 1, state["sms"])
    if stage in ("all", "down"):
        state["tiled_mod"].moe_gemm(state["w2p"], state["gs2"], state["gs2"],
                                    work["inter"], work["y"], work["tstart"], 16, 64, 64, 0, state["sms"])
    if stage in ("all", "finalize"):
        host.combine_tiled_kernel[(t,)](work["y"], work["rows"], fixture["probs"],
                                        state["post_expert"], work["out"], D=H, BNC=128, num_warps=4)
    return work["out"]


def reference(fixture, state, raw):
    x = fixture["x"].float()
    out = torch.zeros_like(x)
    ids, probs = fixture["ids"], fixture["probs"]
    for e in range(E):
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        w13 = nvfp4.dequantize(raw["w13"][e], raw["sf13"][e], 1.)
        # Apply GS after FP32 accumulation, then preserve stock BF16 activations.
        h = x[rows] @ w13.T
        g = (h[:, :H] * state["gs13_gate"][e]).to(torch.bfloat16).float()
        u = (h[:, H:] * state["gs13_up"][e]).to(torch.bfloat16).float()
        inter = (torch.nn.functional.silu(g) * u).to(torch.bfloat16).float()
        w2 = nvfp4.dequantize(raw["w2"][e], raw["sf2"][e], 1.)
        y = ((inter @ w2.T) * state["gs2"][e]).to(torch.bfloat16).float()
        y *= state["post_expert"].float()
        y *= torch.rsqrt((y * y).mean(-1, keepdim=True) + 1e-8)
        out.index_add_(0, rows, probs[rows, slots, None] * y)
    return out.to(torch.bfloat16)


def error_metrics(out, ref):
    diff = out.float() - ref.float()
    den = ref.float().norm(dim=-1)
    floor = (den.mean() * .1).clamp(min=1e-12)
    return dict(relative_l2=float(diff.norm() / ref.float().norm().clamp(min=1e-12)),
                worst_row=float((diff.norm(dim=-1) / den.clamp(min=floor)).max()))


def graph_bench(fn, repeats=3, warmup=1, cold_l2=False):
    global _FLUSH
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    values = []
    if cold_l2 and _FLUSH is None:
        _FLUSH = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for _ in range(repeats):
        if cold_l2:
            _FLUSH.zero_()
        start.record(); graph.replay(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end) * 1000)
    return float(torch.tensor(values).median()), graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", nargs="+", type=int, default=[753, 3012, 16384])
    parser.add_argument("--kinds", nargs="+", default=["rs", "rs128"])
    parser.add_argument("--fixed-n", nargs="+", type=int, default=[0, 128])
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--distribution", default="uniform")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--custom", action="store_true")
    parser.add_argument("--cold-l2", action="store_true")
    parser.add_argument("--json", default=str(HERE / "canonical_comparison.json"))
    args = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    for kind in args.kinds:
        baseline_module(kind)
    custom_mod = custom_host().load_module(str(HERE / "moe_w4a16_canonical.cu")) if args.custom else None
    print("compiled baseline readers", flush=True)
    state, raw = make_weights()
    if custom_mod is not None:
        state["tiled_mod"] = custom_mod
    print("prepared one canonical bank", flush=True)
    result = dict(device=torch.cuda.get_device_name(), architecture=torch.cuda.get_device_capability(),
                  note="Local sm120 MMA.sync emulation; not Hopper WGMMA timing.",
                  source_hashes=json.loads((HERE / "canonical_baselines/SHA256.json").read_text()),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  adapted_grouped_sha256={kind: hashlib.sha256(separate_global_scales(
                      frozen_python(kind).GROUPED_SRC).encode()).hexdigest() for kind in args.kinds},
                  custom_sha256=hashlib.sha256((HERE / "moe_w4a16_canonical.cu").read_bytes()).hexdigest() if args.custom else None,
                  cache_policy="256 MiB flush outside each timing event" if args.cold_l2 else "warm L2, no flush",
                  unequal_gate_up_scales=True, rows=[])
    for tokens in args.tokens:
        fixture = make_fixture(tokens, distribution=args.distribution)
        expected = reference(fixture, state, raw) if args.reference else None
        first_out = None
        for kind in args.kinds + (["pipeline128"] if args.custom else []):
            for width in ([128] if kind == "pipeline128" else args.fixed_n):
                work = custom_workspace(fixture) if kind == "pipeline128" else workspace(fixture)
                run = (lambda: run_custom(fixture, state, work)) if kind == "pipeline128" else (
                    lambda: run_baseline(kind, fixture, state, work, width, args.stages))
                run()
                torch.cuda.synchronize()
                if first_out is None:
                    first_out = work["out"].clone()
                check = error_metrics(work["out"], expected if expected is not None else first_out)
                assert check["relative_l2"] < .015 and check["worst_row"] < .075, check
                timing, graph = graph_bench(run, args.repeats, cold_l2=args.cold_l2)
                # Change x, probabilities and routing in place, then replay the SAME graph.
                fresh = make_fixture(tokens, seed=292, distribution=args.distribution)
                original = {k: v.clone() for k, v in fixture.items()}
                for key in fixture:
                    fixture[key].copy_(fresh[key])
                graph.replay(); torch.cuda.synchronize()
                replay_out = work["out"].clone()
                run(); torch.cuda.synchronize()
                replay_error = error_metrics(replay_out, work["out"])
                assert replay_error["relative_l2"] < 1e-6, replay_error
                if args.reference:
                    fresh_check = error_metrics(replay_out, reference(fixture, state, raw))
                    assert fresh_check["relative_l2"] < .015 and fresh_check["worst_row"] < .075, fresh_check
                else:
                    fresh_check = None
                for key in fixture:
                    fixture[key].copy_(original[key])
                run(); torch.cuda.synchronize()
                stages = {}
                for stage in (("route", "gather", "gate", "down", "finalize") if kind == "pipeline128" else
                              ("route", "gate", "down", "finalize")):
                    runner = (lambda stage=stage: run_custom(fixture, state, work, stage)) if kind == "pipeline128" else (
                        lambda stage=stage: run_baseline(kind, fixture, state, work, width, args.stages, stage))
                    stages[stage], _ = graph_bench(runner, args.repeats, cold_l2=args.cold_l2)
                counts = work["counts"].cpu()
                row = dict(tokens=tokens, kind=kind, fixed_n=width, stages=args.stages,
                           distribution=args.distribution, total_us=timing, stage_us=stages,
                           local_pairs=int(counts.sum()), max_expert_pairs=int(counts.max()),
                           correctness=check, fresh_graph_vs_eager=replay_error,
                           fresh_reference=fresh_check)
                result["rows"].append(row)
                Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(row), flush=True)
                del work, graph, original, fresh
                gc.collect()
        del fixture, first_out, expected
        gc.collect(); torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
