"""Per-problem bounds on one H200, for every scored shape (upstream/api/problems/<name>.json):

- floor:    the judge's floor, recomputed (and checked against TASK.md's table). A lower bound:
            it counts 75% of the expected experts touched and the shortest context share.
- expected: the same data/math, but the expected work (all experts touched, mean context).
- practical: expected bytes at the achievable bandwidth, plus a fixed cost per kernel, for a
            given number of kernels (sim.hw.EFF). `kernels` is the design being priced.

A problem's speedup is the geometric mean over loads of SGLang's load time over ours, and SGLang's
published time (sglang_us) is its load times averaged the same way, so a bound's speedup ceiling
is sglang_us / geomean over loads of the bound's load times."""

import functools
import json
import math
import re
from pathlib import Path

from sim import hw, muse
from sim.routing import expected as expected_routing

REPO = Path(__file__).resolve().parent.parent
ROUTING_SHARE = 0.75        # the judge's floors count this share of the expected experts
AGENTIC_SHORTEST = 0.925    # attention/decode floors: agentic lengths are >= 92.5% of ctx


def spec(problem: str) -> dict:
    return json.loads((REPO / "upstream" / "api" / "problems" / f"{problem}.json").read_text())


def task_floors(problem: str) -> list[float]:
    """The floor column of TASK.md's shape table, in benchmark order."""
    text = (REPO / "problems" / problem / "TASK.md").read_text()
    return [float(v.replace(",", "")) for v in re.findall(r"^\|[^|\n]+\|[^|\n]+\|\s*\d+\s*\|\s*([\d,.]+) µs\s*\|$", text, re.M)]


# --- bytes and flops of one call, per problem -------------------------------------------------
# Each returns (bytes, flops) for the floor-style count; share/touched pick floor vs expected.

PROJ = {"qkv_gate": (muse.QKV_SIZE, muse.HIDDEN), "o": (muse.HIDDEN, muse.Q_SIZE),
        "pooler": (muse.D_MOE, muse.HIDDEN), "unpooler": (muse.HIDDEN, muse.D_MOE),
        "router": (muse.N_EXPERTS, muse.HIDDEN)}


def proj_work(b: dict) -> tuple[float, float]:
    m, layer = b["m"], b["layer"]
    n, k = PROJ[layer]
    if layer == "unpooler" and m <= muse.SLICED_MAX_TOKENS:
        n = muse.UNPOOL_SLICE
    wb = 4 if layer == "router" else 2
    return n * k * wb + m * k * 2 + m * n * wb, 2.0 * m * n * k


def attention_work(b: dict, share: float) -> tuple[float, float]:
    batch, q_len = b["batch"], b.get("q_len", 1)
    attended = min(b["ctx"], muse.WINDOW) if b["layer"] == "sliding" else b["ctx"]
    tokens = batch * attended * share
    io = batch * q_len * (2 * muse.HQ + 2 * muse.HKV) * muse.HEAD_DIM * 2 * 2
    return tokens * muse.KV_BYTES_PER_TOKEN + io, 4.0 * tokens * q_len * muse.HQ * muse.HEAD_DIM


def moe_work(tokens: int, touched: float, routes: float) -> tuple[float, float]:
    per_expert = 3 * muse.D_MOE * muse.D_MOE
    return (touched * muse.EXPERT_BYTES + tokens * muse.D_MOE * 2 * 2, 2.0 * routes * per_expert)


def layer_work(b: dict, touched: float, routes: float, share: float) -> tuple[float, float]:
    """muse_decode_layer / muse_prefill_layer: the reference's speed_of_light_ns with the
    experts and context share given."""
    batch, q_len = b["batch"], b.get("q_len", 1)
    tokens = batch * q_len
    dense_w = muse.dense_bytes(tokens) - muse.ROUTER_BYTES   # bf16 dense weights
    kv_tokens = batch * (min(b["ctx"], muse.WINDOW) if b["layer"] == "sliding" else b["ctx"]) * share
    nbytes = (dense_w + muse.ROUTER_BYTES + touched * muse.EXPERT_BYTES
              + kv_tokens * muse.KV_BYTES_PER_TOKEN + tokens * muse.HIDDEN * (4 + 2) * 2)
    flops = (2.0 * tokens * (dense_w / 2 + muse.N_EXPERTS * muse.HIDDEN) + 2.0 * routes * 3 * muse.D_MOE ** 2
             + 4.0 * kv_tokens * q_len * muse.Q_SIZE)
    return nbytes, flops


# --- the three bounds -------------------------------------------------------------------------

def shape_bounds(problem: str, b: dict, kernels: int, task_floor: float | None = None) -> dict:
    """{'floor', 'expected', 'practical'} in microseconds for one benchmark shape."""
    tokens = b.get("bs") or b.get("m") or b.get("batch", 1) * b.get("q_len", 1)
    distinct, pairs = expected_routing(tokens)
    spread_share = AGENTIC_SHORTEST if b.get("spread") == "agentic" else 1.0
    if problem == "muse_bf16_proj":
        lo = hi = proj_work(b)
    elif problem == "muse_attention_decode":
        lo, hi = attention_work(b, spread_share), attention_work(b, 1.0)
    elif problem == "muse_moe_w4a16":
        lo = moe_work(tokens, max(1.0, ROUTING_SHARE * distinct), max(tokens * 8 * 32 / 256, ROUTING_SHARE * pairs))
        hi = moe_work(tokens, distinct, pairs)
    else:
        # The layer problems route with their own skewed router (task.yml's router_load and
        # router_distinct aren't published), so they are backed out of TASK.md's floor: as experts
        # touched when that floor is the data (at most the 32 local), else as routes (the math).
        # The MoE problem's routing is the fallback.
        if task_floor is None and tokens <= 256:  # not a scored shape: the scored shapes' routing
            distinct, pairs = layer_routing(tokens)
        floor_touched = max(1 if tokens < 16 else 2 if tokens < 128 else 6, ROUTING_SHARE * distinct)
        floor_routes = max(tokens, ROUTING_SHARE * pairs)
        if task_floor is not None:
            rest_bytes, _ = layer_work(b, 0.0, floor_routes, spread_share)
            touched = (task_floor * 1e-6 * hw.HBM_BPS - rest_bytes) / muse.EXPERT_BYTES
            if floor_touched * 0.5 < touched <= muse.LOCAL_EXPERTS:
                floor_touched, distinct = touched, min(touched / ROUTING_SHARE, muse.LOCAL_EXPERTS)
            else:
                floor_touched = distinct = min(distinct, muse.LOCAL_EXPERTS)
                _, rest_flops = layer_work(b, floor_touched, 0.0, spread_share)
                floor_routes = (task_floor * 1e-6 * hw.BF16_FLOPS - rest_flops) / (2.0 * 3 * muse.D_MOE ** 2)
                pairs = floor_routes / ROUTING_SHARE
        lo = layer_work(b, floor_touched, floor_routes, spread_share)
        hi = layer_work(b, distinct, pairs, 1.0)
    floor, expected = hw.floor_us(*lo), hw.floor_us(*hi)
    compute_us = hi[1] / hw.BF16_FLOPS * 1e6
    practical = hw.kernels_us(kernels) + max(hw.stream_us(hi[0]), compute_us / 0.75)
    return {"floor": floor, "expected": expected, "practical": practical, "touched": distinct, "routes": pairs}


def load_times(problem: str, kernels: int) -> dict[str, dict[str, float]]:
    """Each load's weighted time under each bound."""
    loads: dict[str, dict[str, float]] = {}
    for b, tf in zip(spec(problem)["benchmarks"], task_floors(problem)):
        acc = loads.setdefault(b["load"], {"floor": 0.0, "expected": 0.0, "practical": 0.0})
        bounds = shape_bounds(problem, b, kernels, tf)
        for k in acc:
            acc[k] += b["weight"] * bounds[k]
    return loads


@functools.cache
def _layer_routing_table() -> tuple[tuple[int, float, float], ...]:
    rows: dict[int, list[tuple[float, float]]] = {}
    for b, tf in zip(spec("muse_decode_layer")["benchmarks"], task_floors("muse_decode_layer")):
        r = shape_bounds("muse_decode_layer", b, 1, tf)
        rows.setdefault(b["batch"] * b.get("q_len", 1), []).append((r["touched"], r["routes"]))
    return tuple((t, sum(d for d, _ in v) / len(v), sum(p for _, p in v) / len(v)) for t, v in sorted(rows.items()))


def layer_routing(tokens: float) -> tuple[float, float]:
    """(expected experts touched, expected routes) on the busiest GPU under the decode layer's own
    router, as its judge assumes: backed out of its scored shapes, interpolated in log(tokens)."""
    table = _layer_routing_table()
    if tokens <= table[0][0]:
        return table[0][1], table[0][2] * tokens / table[0][0]
    for (t0, d0, p0), (t1, d1, p1) in zip(table, table[1:]):
        if tokens <= t1:
            w = (math.log(tokens) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return d0 + w * (d1 - d0), p0 + w * (p1 - p0)
    t, d, p = table[-1]
    return d, p * tokens / t


def geomean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))
