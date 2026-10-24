"""Prepare-time MoE tuner for decode-layer routing.

It reconstructs the layer's router distribution from the loaded router, bias, pre-FFN norm and
pooler weights. It never reads a submission input: callers provide only their loaded weights and
shape list, so the resulting shape dispatch remains CUDA-graph safe.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Callable

import torch

HIDDEN = 8192
D_EXPERT = 4096
N_EXPERTS = 256
LOCAL_EXPERTS = 32
TOP_K = 8
ROUTER_MULTIPLIER = 0.1767766952966369
RMS_EPS = 1e-5
STREAM_SEED = 5354
STREAM_MEAN = 0.7
STREAM_SHARED = 0.9
STREAM_RANK = 4
DRAFT_SHARE = 0.6


def _rms(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + RMS_EPS)


def _basis(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen).to(device)
    shared = torch.randn((STREAM_RANK, HIDDEN), generator=gen).to(device)
    return mean, shared


def make_fixture(weights: dict[str, torch.Tensor], tokens: int, seed: int, *, mode: str = "real",
                 q_len: int | None = None) -> dict:
    """Return an expert-call fixture with real router or controlled route IDs.

    `q_len` defaults to 8 for the verify loads (8/128/256 tokens), otherwise 1.
    Pass it explicitly for other loads. Draft sharing applies within every verify block.
    `real` uses the layer router and chooses the busiest 32-expert range. `matched_uniform` keeps
    the real local/remote mask but replaces local expert IDs uniformly. `uniform` routes every
    slot to a uniformly chosen local expert, matching older synthetic tuners.
    """
    device = weights["router"].device
    gen = torch.Generator(device=device).manual_seed(seed)
    mean, shared = _basis(device)
    coef = torch.randn((tokens, STREAM_RANK), generator=gen, device=device)
    q_len = (8 if tokens in (8, 128, 256) else 1) if q_len is None else int(q_len)
    if q_len < 1 or tokens % q_len:
        raise ValueError(f"tokens={tokens} must be divisible by q_len={q_len}")
    scale = torch.ones((tokens // q_len, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (STREAM_MEAN * mean + math.sqrt(STREAM_SHARED / STREAM_RANK) * (coef @ shared))
              + (1 - STREAM_SHARED * scale.square()).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    h = _rms(stream) * (1.0 + weights["pre_ffn_norm"].float())
    router = weights["router"].float()
    biased = torch.sigmoid((h @ router.t()) * ROUTER_MULTIPLIER) + weights["router_bias"].float()
    scores = torch.sigmoid((h @ router.t()) * ROUTER_MULTIPLIER)
    global_ids = biased.topk(TOP_K, dim=-1).indices.to(torch.int32)
    picked = scores.gather(1, global_ids.long())
    topk_weights = picked / picked.sum(-1, keepdim=True).clamp_min(1e-15)
    ranges = torch.bincount((global_ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    local_start = LOCAL_EXPERTS * int(ranges.argmax())
    local = ((global_ids - local_start) % N_EXPERTS).to(torch.int32)
    local_mask = local < LOCAL_EXPERTS
    if mode == "uniform":
        lids = torch.randint(0, LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
    elif mode == "matched_uniform":
        lids = torch.randint(0, LOCAL_EXPERTS, global_ids.shape, generator=gen, device=device, dtype=torch.int32)
        lids = torch.where(local_mask, lids, torch.full_like(lids, -1))
    elif mode == "real":
        lids = torch.where(local_mask, local, torch.full_like(local, -1))
    else:
        raise ValueError(mode)
    # Pooler output is the expert input in the decode layer. Keep it identical across route modes.
    x = h @ weights["pooler"].float().t()
    x = (_rms(x) * (1.0 + weights["pre_expert_norm"].float())).to(torch.bfloat16)
    tw = topk_weights if mode != "uniform" else torch.rand((tokens, TOP_K), generator=gen, device=device)
    if mode == "uniform":
        tw = tw / tw.sum(-1, keepdim=True)
    cfg = {"local_count": 32, "d_hidden": D_EXPERT, "d_expert": D_EXPERT, "local_start": 0}
    return {"hidden_states": x, "topk_ids": lids, "topk_weights": tw.float(), "config": cfg,
            "local_start": local_start, "global_ids": global_ids, "local_mask": local_mask}


def route_summary(fixture: dict) -> dict[str, float]:
    ids = fixture["topk_ids"]
    local = ids[ids >= 0]
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS)
    return {"pairs": float(local.numel()), "distinct": float((counts > 0).sum()),
            "max_expert": float(counts.max()), "mean_nonzero": float(counts[counts > 0].float().mean())}


def _graph_time(fn: Callable[[dict], torch.Tensor], datasets: list[dict], flush: torch.Tensor, repeats: int = 4) -> float:
    first = datasets[0]
    static = {k: first[k].clone() if torch.is_tensor(first[k]) else first[k] for k in first if k not in ("global_ids", "local_mask", "local_start")}
    for _ in range(2):
        fn(static)
    torch.cuda.synchronize(first["hidden_states"].device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(static)
    torch.cuda.synchronize(first["hidden_states"].device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vals = []
    for r in range(repeats):
        fresh = datasets[r % len(datasets)]
        for k in ("hidden_states", "topk_ids", "topk_weights"):
            static[k].copy_(fresh[k])
        flush.sum()
        start.record(); graph.replay(); end.record(); end.synchronize()
        vals.append(float(start.elapsed_time(end)))
    del graph, static
    torch.cuda.empty_cache()
    return statistics.median(vals)


def tune(weights: dict[str, torch.Tensor], paths: dict[str, Callable[[dict], torch.Tensor]],
         token_counts=(1, 8, 16, 32, 128, 256), *, seed: int = 5353,
         flush_bytes: int = 1 << 30, budget_s: float = 90.0) -> tuple[dict[int, str], dict[int, dict[str, float]], dict[int, dict]]:
    """Rank paths on this layer's router and keep only paths within 1% of V9.

    The timer is CUDA-graph replay after an L2 flush.  A wall-clock deadline protects the
    layer's overall prepare limit; any shape not reached remains on V9.  A bad optional path
    is rejected per shape, while V9 is always the fallback.
    """
    device = weights["router"].device
    deadline = time.monotonic() + float(budget_s)
    flush = torch.empty((flush_bytes,), dtype=torch.uint8, device=device)
    choices, timings, summaries = {}, {}, {}
    for m in token_counts:
        if time.monotonic() >= deadline:
            for rest in token_counts:
                if rest not in choices:
                    choices[rest] = "v9"; timings[rest] = {"v9": 0.0}
                    print(f"MOE_REAL_BUDGET M={rest} selected=v9", flush=True)
            break
        datasets = [make_fixture(weights, m, seed + 7919 * j + m, mode="real") for j in range(3)]
        summaries[m] = route_summary(datasets[0])
        candidates = {}
        want = None
        try:
            want = paths["v9"](datasets[0]).float()
            if not torch.isfinite(want).all().item():
                raise RuntimeError("V9 produced non-finite output")
        except Exception as exc:
            # The layer's V9 path is the canonical fallback. Keep moving through preparation,
            # with the explicit default visible in the receipt instead of hiding the failure.
            print(f"MOE_REAL_V9_ERROR M={m} {str(exc).splitlines()[0][:120]}", flush=True)
            choices[m] = "v9"; timings[m] = {"v9": 0.0}
            del datasets
            continue
        for name, fn in paths.items():
            if time.monotonic() >= deadline and name != "v9":
                print(f"MOE_REAL_BUDGET M={m} path={name}", flush=True)
                continue
            try:
                got = fn(datasets[0]).float()
                rel = float((got - want).norm() / want.norm().clamp_min(1e-12))
                if not torch.isfinite(got).all().item() or rel > 0.01:
                    print(f"MOE_REAL_REJECT M={m} {name} rel={rel:.4g}", flush=True)
                    continue
                candidates[name] = _graph_time(fn, datasets, flush)
            except Exception as exc:
                print(f"MOE_REAL_REJECT M={m} {name} {str(exc).splitlines()[0][:120]}", flush=True)
        if not candidates:
            choices[m] = "v9"; timings[m] = {"v9": 0.0}
            print(f"MOE_REAL_FALLBACK M={m} selected=v9", flush=True)
        else:
            # V9 is in the candidate set whenever its first call succeeded. If its graph
            # replay failed, retain the required V9 default instead of selecting a partial set.
            choices[m] = min(candidates, key=candidates.get) if "v9" in candidates else "v9"
            timings[m] = candidates
        print(f"MOE_REAL_TUNE M={m} route={summaries[m]} choices={candidates} selected={choices[m]}", flush=True)
        del datasets
        torch.cuda.empty_cache()
    del flush
    torch.cuda.empty_cache()
    return choices, timings, summaries
