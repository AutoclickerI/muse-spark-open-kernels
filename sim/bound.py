"""Rules-compliant lower bound for the Muse Spark Open kernel and serving tracks.

This is deliberately more exact than :mod:`sim.problems`' organizer-floor model:

* routing uses the scored distribution's expected work (or the expectation inferred by
  ``sim.problems`` from the official layer floors), rather than the official 75% discount;
* NVFP4 codes, E4M3 block scales, FP32 global scales, dense weights, paged-cache indices,
  compulsory input/output activations, and the router's BF16-hi+lo arithmetic are explicit;
* QK uses the allowed FP8-query x FP8-key peak, while PV and every other matrix product use
  the dense BF16 peak;
* NVFP4 decode has both a theoretical packed-LUT issue floor and docs/reference/profile.md's audited
  1.5-integer-instructions/weight floor; and
* raw graph time adds one measured replay cost.  The competition subtracts this cost, so
  speedup ceilings use the replay-adjusted resource time.

The serving bound shards the otherwise replicated pooler/unpooler exactly, pays the resulting
all-gathers plus the two model all-reduces, uses the two BF16 expert layers, and permits perfect
overlap of aggregate HBM, tensor-core, dequant and NVLink work inside a step.  It is consequently
a physical lower bound, not an implementation forecast.

Run from the repository root::

    python -m sim.bound
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from sim import muse, serving
from sim.calibrate import sglang_mean
from sim.problems import (
    geomean,
    layer_routing,
    load_times,
    shape_bounds,
    spec,
    task_floors,
)
from sim.routing import expected as expected_routing

REPO = Path(__file__).resolve().parent.parent
PROBLEMS = (
    "muse_bf16_proj",
    "muse_attention_decode",
    "muse_moe_w4a16",
    "muse_decode_layer",
    "muse_prefill_layer",
)

# Rates requested by the owner.  NVIDIA publishes twice these tensor rates in the H200 table
# because that column includes 2:1 sparsity; the competition and this model use dense rates.
SPEC_BW = 4.8e12
MEASURED_BW = 4.25e12
BF16 = 989e12
FP8 = 1979e12

# runs/muse_bf16_proj/*7fb3e93056d34999.json, first benchmark record:
# replay_overhead.overhead_ns = 4,224.000032991171.  The diagnostic graph launcher reports
# 4.93--5.31 us in the cand_sweep/diag logs, but 4.224 us is the evaluator's actual subtraction.
REPLAY_US = 4.224000033

# H200 measured state in the same receipt: 132 SMs at 1.980 GHz.  docs/reference/hardware.md cites 64 integer /
# logic issues per SM-cycle.  A 256-entry packed LUT can map one byte (two E2M1 codes) to f16x2
# in one load/permute, hence 0.5 issue instructions/weight is the widest defensible theoretical
# minimum.  docs/reference/profile.md's static source/SASS audit is about 1.5 integer instructions/weight.
INT_ISSUES_PER_S = 132 * 1.980e9 * 64
DEQUANT_THEORY_INST_PER_WEIGHT = 0.5
DEQUANT_AUDIT_INST_PER_WEIGHT = 1.5

# NVLink 4 is 900 GB/s bidirectional = 450 GB/s in each direction.  Third-generation NVSwitch
# supplies multicast and SHARP/NVLS reductions.  docs/reference/hardware.md records the best published 8xH200
# small-collective measurement as 2.4 us; production NVLS is typically 5.6--5.9 us.
NVLINK_ONE_WAY_BPS = 450e9
COLLECTIVE_LATENCY_US = 2.4

# Snapshot values fetched from the public API on 2026-10-23.  Keeping the bound runnable offline
# matters more than making it silently depend on a moving leaderboard.
LEADERS = {
    "muse_bf16_proj": 1.4466332928118113,
    "muse_attention_decode": 2.3659224609983576,
    "muse_moe_w4a16": 2.325909167844873,
    "muse_decode_layer": 1.8642447656123302,
    "muse_prefill_layer": 1.695434412925069,
}


@dataclass(frozen=True)
class Work:
    """Compulsory work on one GPU.

    ``bf16_flops`` and ``fp8_flops`` share the tensor cores, so their peak times add.  Integer
    dequant can overlap those tensor operations and HBM movement, so it is a separate max term.
    ``scalar_ops`` is reported but intentionally has no numeric floor: exp/rsqrt/top-k instruction
    sequences are implementation-dependent and no H200 peak for the required mix is published.
    """

    nbytes: float = 0.0
    bf16_flops: float = 0.0
    fp8_flops: float = 0.0
    nvfp4_weights: float = 0.0
    scalar_ops: float = 0.0

    def __add__(self, other: Work) -> Work:
        return Work(*(a + b for a, b in zip(self.values(), other.values())))

    def __sub__(self, other: Work) -> Work:
        return Work(*(a - b for a, b in zip(self.values(), other.values())))

    def __mul__(self, factor: float) -> Work:
        return Work(*(factor * a for a in self.values()))

    __rmul__ = __mul__

    def values(self) -> tuple[float, float, float, float, float]:
        return self.nbytes, self.bf16_flops, self.fp8_flops, self.nvfp4_weights, self.scalar_ops


@dataclass(frozen=True)
class Floor:
    memory_us: float
    tensor_us: float
    scalar_us: float
    dequant_theory_us: float
    dequant_audit_us: float
    communication_us: float
    adjusted_us: float
    raw_us: float
    bottleneck: str


def floor(work: Work, bandwidth: float, communication_us: float = 0.0, replays: float = 1.0) -> Floor:
    memory = work.nbytes / bandwidth * 1e6
    # Both precisions issue on the same tensor-core resource; QK and dependent PV cannot receive
    # both peaks simultaneously, so peak-equivalent times add rather than taking another max.
    tensor = (work.bf16_flops / BF16 + work.fp8_flops / FP8) * 1e6
    # Give every counted non-matrix operation the GPU's *entire* dense-FP8 rate.  This is far more
    # generous than the CUDA/SFU pipes can achieve, hence remains a valid lower floor without
    # pretending that exp/rsqrt/top-k have a published common throughput.
    scalar = work.scalar_ops / FP8 * 1e6
    dq_min = work.nvfp4_weights * DEQUANT_THEORY_INST_PER_WEIGHT / INT_ISSUES_PER_S * 1e6
    dq_audit = work.nvfp4_weights * DEQUANT_AUDIT_INST_PER_WEIGHT / INT_ISSUES_PER_S * 1e6
    resources = {"HBM": memory, "tensor": tensor, "scalar/SFU": scalar,
                 "NVFP4 issue (audited)": dq_audit,
                 "NVLink": communication_us}
    bottleneck = max(resources, key=resources.get)
    adjusted = resources[bottleneck]
    return Floor(memory, tensor, scalar, dq_min, dq_audit, communication_us, adjusted,
                 adjusted + replays * REPLAY_US, bottleneck)


def nvfp4_expert_work(touched: float, routes: float, activation_tokens: float,
                      activation_factor: float = 1.0, scale: float = 1.0,
                      external_io: bool = True) -> Work:
    """One local routed-expert block, including exact NVFP4 format metadata and external I/O."""
    touched, routes = touched * scale, routes * scale
    weights_per_expert = 3 * muse.D_MOE * muse.D_MOE
    # 4-bit codes + one E4M3 byte/16 weights + two FP32 global scales/expert.  post_norm_w is
    # shared by all experts and read once.  Inputs include all top-8 IDs and FP32 route weights.
    weight_bytes = touched * (weights_per_expert * (0.5 + 1 / 16) + 2 * 4)
    io = activation_tokens * muse.D_MOE * 2 * 2 * activation_factor if external_io else 0.0
    routing = activation_tokens * muse.TOP_K * (4 + 4) if external_io else 0.0
    vectors = muse.D_MOE * 2
    gemm_flops = 2.0 * routes * weights_per_expert
    # A conservative arithmetic count for SwiGLU, per-expert RMS, scaling and accumulation.
    scalar = routes * muse.D_MOE * 12
    return Work(weight_bytes + io + routing + vectors, gemm_flops, 0.0,
                touched * weights_per_expert, scalar)


def bf16_expert_work(touched: float, routes: float, activation_tokens: float,
                     activation_factor: float = 1.0, scale: float = 1.0,
                     external_io: bool = True) -> Work:
    """The same routed block with the first/last layer's lossless BF16 expert weights."""
    touched, routes = touched * scale, routes * scale
    weights_per_expert = 3 * muse.D_MOE * muse.D_MOE
    io = activation_tokens * muse.D_MOE * 2 * 2 * activation_factor if external_io else 0.0
    routing = activation_tokens * muse.TOP_K * (4 + 4) if external_io else 0.0
    vectors = muse.D_MOE * 2
    return Work(touched * weights_per_expert * 2 + io + routing + vectors,
                2.0 * routes * weights_per_expert, 0.0, 0.0,
                routes * muse.D_MOE * 12)


def attention_work(batch: int, q_len: int, ctx: float, layer: str, *, external_io: bool,
                   hq: int = muse.HQ, hkv: int = muse.HKV, head_dim: int = muse.HEAD_DIM,
                   kv_element_bytes: int = 1, activation_factor: float = 1.0,
                   attention_cap: int | None = None) -> Work:
    """Exact ideal-reuse attention work.

    Every old KV row is loaded once and reused for every query in the block.  New K/V stays in
    registers for attention but must be written to the cache.  Page size one requires one int32
    page-table entry per old/new logical row.  QK is the sole legal FP8 tensor operation; PV keeps
    its BF16 softmax operand and is charged at BF16 peak.
    """
    tokens = batch * q_len
    if attention_cap is not None:
        # Counterfactual only: attend at most this many keys, irrespective of full/sliding type.
        per_query = min(ctx + (q_len + 1) / 2, attention_cap)
        interactions = batch * q_len * per_query
        old_unique = batch * min(ctx, attention_cap)
    elif layer == "sliding":
        per_query = min(ctx + (q_len + 1) / 2, muse.WINDOW)
        interactions = batch * q_len * per_query
        old_unique = batch * min(ctx, max(0, muse.WINDOW - 1))
    else:
        interactions = batch * (q_len * ctx + q_len * (q_len + 1) / 2)
        old_unique = batch * ctx

    kv_row = 2 * hkv * head_dim * kv_element_bytes
    cache_bytes = old_unique * kv_row + tokens * kv_row
    page_bytes = (old_unique + tokens) * 4 + batch * 4  # page table plus seq_lens
    io = 0.0
    if external_io:
        qkv_width = 2 * hq * head_dim + 2 * hkv * head_dim
        io = tokens * (qkv_width * 2 + hq * head_dim * 2) * activation_factor
    qk = 2.0 * interactions * hq * head_dim
    pv = qk
    # max/subtract/exp/sum/divide per score, plus Q/K norm/gate/RoPE bookkeeping.
    scalar = interactions * hq * 5 + tokens * (4 * hq + 4 * hkv) * head_dim
    return Work(cache_bytes + page_bytes + io, pv, qk, 0.0, scalar)


PROJ = {
    "qkv_gate": (muse.QKV_SIZE, muse.HIDDEN),
    "o": (muse.HIDDEN, muse.Q_SIZE),
    "pooler": (muse.D_MOE, muse.HIDDEN),
    "unpooler": (muse.HIDDEN, muse.D_MOE),
    "router": (muse.N_EXPERTS, muse.HIDDEN),
}


def projection_work(case: dict, activation_factor: float = 1.0) -> Work:
    m, name = case["m"], case["layer"]
    n, k = PROJ[name]
    if name == "unpooler" and m <= muse.SLICED_MAX_TOKENS:
        n = muse.UNPOOL_SLICE
    element_bytes = 4 if name == "router" else 2
    nbytes = n * k * element_bytes + m * k * 2 * activation_factor + m * n * element_bytes * activation_factor
    # Lossless FP32 router repacking is two BF16 matrices (hi + lo), each accumulated in FP32.
    flops = (4.0 if name == "router" else 2.0) * m * n * k
    return Work(nbytes, flops)


def inferred_routing(problem: str, case: dict, task_floor: float) -> tuple[float, float]:
    tokens = case.get("bs") or case.get("batch", 1) * case.get("q_len", 1)
    if problem == "muse_moe_w4a16":
        return expected_routing(tokens)
    r = shape_bounds(problem, case, 1, task_floor)
    return r["touched"], r["routes"]


def layer_work(case: dict, touched: float, routes: float, *, shard_replicated: bool = False,
               bf16_expert_share: float = 0.0, activation_factor: float = 1.0,
               attention_cap: int | None = None, expert_scale: float = 1.0) -> Work:
    batch, q_len = case["batch"], case.get("q_len", 1)
    tokens = batch * q_len
    unpool_n = muse.UNPOOL_SLICE if tokens <= muse.SLICED_MAX_TOKENS else muse.HIDDEN
    pool_elems = muse.D_MOE * muse.HIDDEN
    unpool_elems = unpool_n * muse.D_MOE
    if shard_replicated:
        pool_elems /= muse.TP
        # In serving we shard the full unpooler even above eight tokens.  At <=8, the kernel
        # problem already describes one rank's 1024-row slice.
        unpool_elems = muse.HIDDEN * muse.D_MOE / muse.TP
    dense_bf16_elems = (muse.QKV_SIZE * muse.HIDDEN + muse.HIDDEN * muse.Q_SIZE
                        + pool_elems + unpool_elems)
    router_elems = muse.N_EXPERTS * muse.HIDDEN
    router_bias_bytes = muse.N_EXPERTS * 4
    if shard_replicated:
        # Exact output-row sharding: each rank computes 32 logits, then all-gathers all 256 for
        # the global top-8.  This also losslessly shards both BF16 halves of the FP32 router.
        router_elems /= muse.TP
        router_bias_bytes /= muse.TP

    # Seven BF16 norm/gate vectors plus the FP32 router bias.  The exact vector widths are from
    # reference.generate_weights; each can be streamed once in a fully fused layer.
    vector_bytes = (5 * muse.HIDDEN + 2 * muse.D_MOE) * 2 + router_bias_bytes
    dense = Work(dense_bf16_elems * 2 + router_elems * 4 + vector_bytes,
                 2.0 * tokens * dense_bf16_elems + 4.0 * tokens * router_elems,
                 scalar_ops=tokens * (18 * muse.HIDDEN + 10 * muse.D_MOE + muse.N_EXPERTS))

    # Pooler output, routing and expert output are internal to the layer.  An ideal fused
    # ordering need not spill them to HBM; counting them would not be a valid lower bound.
    nv = nvfp4_expert_work(touched, routes, tokens, activation_factor, expert_scale,
                           external_io=False)
    bf = bf16_expert_work(touched, routes, tokens, activation_factor, expert_scale,
                          external_io=False)
    experts = nv * (1 - bf16_expert_share) + bf * bf16_expert_share

    attn = attention_work(batch, q_len, case["ctx"], case["layer"], external_io=False,
                          activation_factor=activation_factor, attention_cap=attention_cap)
    # Fresh input stream + attention input, final stream + next hidden + returned expert IDs.
    io = tokens * (muse.HIDDEN * (4 + 2) * 2 + muse.TOP_K * 4) * activation_factor
    return dense + experts + attn + Work(io)


def problem_shape_work(problem: str, case: dict, task_floor: float) -> Work:
    if problem == "muse_bf16_proj":
        return projection_work(case)
    if problem == "muse_attention_decode":
        return attention_work(case["batch"], case.get("q_len", 1), case["ctx"], case["layer"],
                              external_io=True)
    touched, routes = inferred_routing(problem, case, task_floor)
    if problem == "muse_moe_w4a16":
        return nvfp4_expert_work(touched, routes, case["bs"])
    return layer_work(case, touched, routes)


def stock_loads(problem: str) -> dict[str, float]:
    """Per-load replay-adjusted SGLang time from the mean of local H200 receipts."""
    rows = sglang_mean(problem)
    benches = spec(problem)["benchmarks"]
    if len(rows) != len(benches):
        raise RuntimeError(f"{problem}: {len(rows)} calibration rows for {len(benches)} benchmarks")
    out: dict[str, float] = {}
    for b, r in zip(benches, rows):
        if b["load"] != r["case"]["load"]:
            raise RuntimeError(f"{problem}: calibration order no longer matches official benchmarks")
        out[b["load"]] = out.get(b["load"], 0.0) + b["weight"] * r["sglang_us"]
    return out


def problem_bound(problem: str) -> dict:
    benches, official_floors = spec(problem)["benchmarks"], task_floors(problem)
    stock = stock_loads(problem)
    loads: dict[str, dict] = {}
    issue_binds = False
    for case, task_floor in zip(benches, official_floors):
        work = problem_shape_work(problem, case, task_floor)
        fs, fm = floor(work, SPEC_BW), floor(work, MEASURED_BW)
        issue_binds |= fs.dequant_audit_us >= max(fs.memory_us, fs.tensor_us)
        row = loads.setdefault(case["load"], {"work_spec": 0.0, "work_measured": 0.0,
                                              "raw_spec": 0.0, "raw_measured": 0.0,
                                              "bytes": 0.0, "bf16": 0.0, "fp8": 0.0,
                                              "nvweights": 0.0, "calls": 0.0})
        weight = case["weight"]
        row["work_spec"] += weight * fs.adjusted_us
        row["work_measured"] += weight * fm.adjusted_us
        row["raw_spec"] += weight * fs.raw_us
        row["raw_measured"] += weight * fm.raw_us
        row["bytes"] += weight * work.nbytes
        row["bf16"] += weight * work.bf16_flops
        row["fp8"] += weight * work.fp8_flops
        row["nvweights"] += weight * work.nvfp4_weights
        row["calls"] += weight
    for name, row in loads.items():
        row["stock"] = stock[name]
        row["speedup_spec"] = row["stock"] / row["work_spec"]
        row["speedup_measured"] = row["stock"] / row["work_measured"]
    gm = {}
    for key in ("work_spec", "work_measured", "raw_spec", "raw_measured"):
        gm[key] = geomean(r[key] for r in loads.values())
    gm["paired_stock"] = geomean(r["stock"] for r in loads.values())
    # Use the organizers' published SGLang geomean for the headline, as sim.__main__ and
    # docs/reference/bound.md do.  Per-load ratios above remain tied to our paired calibration receipts.
    gm["stock"] = float(spec(problem)["sglang_us"])
    gm["speedup_spec"] = gm["stock"] / gm["work_spec"]
    gm["speedup_measured"] = gm["stock"] / gm["work_measured"]
    kernels = {"muse_bf16_proj": 1, "muse_attention_decode": 2, "muse_moe_w4a16": 2,
               "muse_decode_layer": 8, "muse_prefill_layer": 10}[problem]
    practical_loads = load_times(problem, kernels)
    practical_us = geomean(v["practical"] for v in practical_loads.values())
    return {"loads": loads, "gm": gm, "issue_binds": issue_binds,
            "practical_us": practical_us,
            "practical_speedup": float(spec(problem)["sglang_us"]) / practical_us}


def allreduce_us(payload: float) -> float:
    """NVLS lower floor: one full payload injected and returned on simultaneous directions."""
    return COLLECTIVE_LATENCY_US + payload / NVLINK_ONE_WAY_BPS * 1e6


def allgather_us(full_payload: float) -> float:
    """Each TP8 rank receives seven eighths of a row-sharded full output."""
    return COLLECTIVE_LATENCY_US + (7 / 8) * full_payload / NVLINK_ONE_WAY_BPS * 1e6


def layer_communication_us(tokens: int, *, shard_replicated: bool = True) -> float:
    # Attention row-parallel output and EP expert output are all-reduced.  Exact pooler and
    # unpooler row sharding add all-gathers of the complete BF16 outputs.
    total = allreduce_us(tokens * muse.HIDDEN * 2) + allreduce_us(tokens * muse.D_MOE * 2)
    if shard_replicated:
        # Router and pooler are independent branches with the same join point.  Pack their
        # row-shards into one exact all-gather to pay a single startup floor.
        total += (allgather_us(tokens * (muse.D_MOE * 2 + muse.N_EXPERTS * 4))
                  + allgather_us(tokens * muse.HIDDEN * 2))
    return total


def target_step(c: int, *, activation_factor: float = 1.0, attention_cap: int | None = None,
                expert_scale: float = 1.0) -> tuple[Work, float]:
    tokens = c * serving.BLOCK
    touched, routes = layer_routing(tokens)
    mix = serving.BF16_EXPERT_LAYERS / muse.LAYERS
    work = Work()
    comm = 0.0
    for layer, count in (("sliding", muse.SLIDING_LAYERS), ("full", muse.FULL_LAYERS)):
        case = {"batch": c, "q_len": serving.BLOCK, "ctx": serving.CTX, "layer": layer}
        work += count * layer_work(case, touched, routes, shard_replicated=True,
                                   bf16_expert_share=mix, activation_factor=activation_factor,
                                   attention_cap=attention_cap, expert_scale=expert_scale)
        comm += count * layer_communication_us(tokens)
    return work, comm


def drafter_step(c: int, *, activation_factor: float = 1.0, free: bool = False) -> tuple[Work, float]:
    if free:
        return Work(), 0.0
    d, tokens = muse.DRAFT, c * serving.BLOCK
    # All standard attention/MLP matrices are TP8.  The 5-target-layer context FC is exactly
    # column-sharded for the lower bound (the released BF16 values are unchanged), then gathered.
    attn_elems = d["hidden"] * (2 * d["heads"] + 2 * d["kv_heads"]) * d["head_dim"]
    mlp_elems = 3 * d["hidden"] * d["inter"]
    fc_elems = d["context_layers"] * d["hidden"] * d["hidden"] / muse.TP
    layer_elems = d["layers"] * (attn_elems + mlp_elems) / muse.TP
    # DFlash sends candidates through the target's vocab-parallel head once.
    head_elems = muse.LM_HEAD_BYTES / 2
    weight_elems = layer_elems + fc_elems + head_elems
    qh, kh, hd = d["heads"] // muse.TP, d["kv_heads"] // muse.TP, d["head_dim"]
    # The draft cache remains BF16 under the serving precision rule.  Five 2K-window layers.
    kv = d["layers"] * c * muse.WINDOW * 2 * kh * hd * 2
    io = tokens * d["hidden"] * 2 * 2 * activation_factor
    logits = tokens * (muse.VOCAB // muse.TP) * 2 * activation_factor
    interactions = d["layers"] * c * serving.BLOCK * muse.WINDOW
    attn_flops = 4.0 * interactions * qh * hd
    work = Work(weight_elems * 2 + kv + io + logits,
                2.0 * tokens * weight_elems + attn_flops,
                scalar_ops=interactions * qh * 5)
    payload = tokens * d["hidden"] * 2
    comm = 2 * d["layers"] * allreduce_us(payload) + allgather_us(payload)
    return work, comm


def lm_head_step(c: int, activation_factor: float = 1.0) -> Work:
    tokens = c * serving.BLOCK
    elems = muse.LM_HEAD_BYTES / 2
    io = tokens * (muse.HIDDEN + muse.VOCAB // muse.TP) * 2 * activation_factor
    return Work(muse.LM_HEAD_BYTES + io, 2.0 * tokens * elems)


def serving_step(c: int, bandwidth: float, *, activation_factor: float = 1.0,
                 attention_cap: int | None = None, expert_scale: float = 1.0,
                 free_drafter: bool = False) -> dict:
    target, target_comm = target_step(c, activation_factor=activation_factor,
                                      attention_cap=attention_cap, expert_scale=expert_scale)
    draft, draft_comm = drafter_step(c, activation_factor=activation_factor, free=free_drafter)
    head = lm_head_step(c, activation_factor)
    total = target + draft + head
    f = floor(total, bandwidth, target_comm + draft_comm)
    return {"floor": f, "work": total, "target": target, "drafter": draft, "lm_head": head,
            "target_comm_us": target_comm, "draft_comm_us": draft_comm}


def prefill_groups(load: str) -> list[list[tuple[dict, float, float]]]:
    """Groups of (case, task_floor, layer count) that form sequential model forwards."""
    rows = [(b, tf) for b, tf in zip(spec("muse_prefill_layer")["benchmarks"],
                                     task_floors("muse_prefill_layer")) if b["load"] == load]
    if load != "first turn":
        return [[(b, tf, float(b["weight"])) for b, tf in rows]]
    sliding = next((b, tf) for b, tf in rows if b["layer"] == "sliding")
    full = [(b, tf) for b, tf in rows if b["layer"] == "full"]
    # The official table folds three identical sliding-window stacks into weight 138.
    return [[(sliding[0], sliding[1], muse.SLIDING_LAYERS), (b, tf, muse.FULL_LAYERS)]
            for b, tf in full]


def prefill_load(load: str, bandwidth: float, *, activation_factor: float = 1.0,
                 attention_cap: int | None = None, expert_scale: float = 1.0) -> dict:
    total_us = 0.0
    aggregate = Work()
    aggregate_comm = 0.0
    bottlenecks: list[str] = []
    mix = serving.BF16_EXPERT_LAYERS / muse.LAYERS
    for group in prefill_groups(load):
        work, comm = Work(), 0.0
        for case, tf, count in group:
            touched, routes = inferred_routing("muse_prefill_layer", case, tf)
            one = layer_work(case, touched, routes, shard_replicated=True,
                             bf16_expert_share=mix, activation_factor=activation_factor,
                             attention_cap=attention_cap, expert_scale=expert_scale)
            work += count * one
            comm += count * layer_communication_us(case["batch"] * case["q_len"])
        f = floor(work, bandwidth, comm)
        total_us += f.raw_us
        aggregate += work
        aggregate_comm += comm
        bottlenecks.append(f.bottleneck)
    return {"raw_us": total_us, "adjusted_us": total_us - len(prefill_groups(load)) * REPLAY_US,
            "work": aggregate, "comm_us": aggregate_comm, "bottlenecks": bottlenecks,
            "forwards": len(prefill_groups(load))}


def stock_and_practical_curves() -> tuple[dict, dict, float]:
    args = SimpleNamespace(tau=4.16, turns=13, host_us=0.0, interactivity="e2e")
    stock = serving.curve("stock", args)
    practical = serving.curve("practical", args)
    practical_score = serving.score(practical, stock, capped=True)[0]
    return stock, practical, practical_score


def bound_curve(bandwidth: float, tau: float, *, activation_factor: float = 1.0,
                attention_cap: int | None = None, expert_scale: float = 1.0,
                free_drafter: bool = False) -> tuple[dict, dict]:
    first = prefill_load("first turn", bandwidth, activation_factor=activation_factor,
                         attention_cap=attention_cap, expert_scale=expert_scale)
    later = prefill_load("turn 1", bandwidth, activation_factor=activation_factor,
                         attention_cap=attention_cap, expert_scale=expert_scale)
    prefill = first["raw_us"] + 12 * later["raw_us"]
    output = 13 * serving.OUTPUT_TOKENS
    curve = {}
    steps = {}
    for c in serving.RUNGS:
        s = serving_step(c, bandwidth, activation_factor=activation_factor,
                         attention_cap=attention_cap, expert_scale=expert_scale,
                         free_drafter=free_drafter)
        step = s["floor"].raw_us
        window = c * prefill + output / tau * step
        curve[c] = {"step_us": step, "per_gpu": output * c / window * 1e6 / 8,
                    "per_user": output / window * 1e6,
                    "prefill_share": c * prefill / window}
        steps[c] = s
    return curve, {"steps": steps, "first": first, "later": later, "conversation_us": prefill}


def serving_results() -> dict:
    stock, practical, practical_score = stock_and_practical_curves()
    # 3.62 is NVIDIA's best published accepted length with no more than the judge's eight-draft
    # budget (seven N-gram drafts on a translation workload).  4.93 is the more relevant but
    # explicitly inferred hybrid envelope described below.
    taus = (3.62, 4.16, 4.93, 9.0)
    rates = {}
    for name, bw in (("spec", SPEC_BW), ("measured", MEASURED_BW)):
        scores = {}
        details = None
        for tau in taus:
            curve, detail = bound_curve(bw, tau)
            ex, per_ex = serving.score(curve, stock, capped=False)
            cap, per_cap = serving.score(curve, stock, capped=True)
            scores[tau] = {"curve": curve, "score_extrapolated": ex, "score_capped": cap,
                           "per_extrapolated": per_ex, "per_capped": per_cap}
            if tau == 4.16:
                details = detail
        rates[name] = {"scores": scores, "details": details}

    # Clearly inadmissible sensitivities, all at spec rates.  Tau=4.93 is an optimistic hybrid
    # envelope: DFlash's measured 3.16 accepted drafts plus NVIDIA's multi-turn N-gram 0.77.
    legal_score = rates["spec"]["scores"][4.16]["score_capped"]
    scenarios = {}
    for name, kwargs, tau in (
        ("FP8 activations", {"activation_factor": 0.5}, 4.16),
        ("FP4 activations", {"activation_factor": 0.25}, 4.16),
        ("2K sparse full attention", {"attention_cap": 2048}, 4.16),
        ("top-2 experts", {"expert_scale": 0.25}, 4.16),
        ("free perfect replacement drafter", {"free_drafter": True}, 9.0),
    ):
        curve, detail = bound_curve(SPEC_BW, tau, **kwargs)
        score = serving.score(curve, stock, capped=True)[0]
        scenarios[name] = {"score": score, "gain": score / legal_score,
                           "step32_us": detail["steps"][32]["floor"].raw_us}
    return {"stock": stock, "practical": practical, "practical_score": practical_score,
            "rates": rates, "scenarios": scenarios}


def best_receipt(problem: str) -> tuple[float, str]:
    candidates = []
    for name in glob.glob(str(REPO / "runs" / problem / "*.json")):
        data = json.loads(Path(name).read_text())
        result = data.get("result") or {}
        if result.get("passed") and not result.get("flagged") and isinstance(result.get("speedup"), (int, float)):
            candidates.append((float(result["speedup"]), str(Path(name).relative_to(REPO))))
    if not candidates:
        return float("nan"), "none"
    return max(candidates)


def fmt_ms(us: float) -> str:
    return f"{us / 1e3:,.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the computed result as JSON")
    args = ap.parse_args()
    problems = {p: problem_bound(p) for p in PROBLEMS}
    serv = serving_results()
    if args.json:
        # Dataclasses appear only below the serving step details; make them serializable.
        def default(obj):
            if isinstance(obj, (Work, Floor)):
                return obj.__dict__
            raise TypeError(type(obj).__name__)
        print(json.dumps({"problems": problems, "serving": serv}, default=default, indent=2))
        return

    print(f"H200: HBM {SPEC_BW/1e12:.2f} TB/s spec / {MEASURED_BW/1e12:.2f} TB/s measured; "
          f"BF16 {BF16/1e12:.0f}, FP8 QK {FP8/1e12:.0f} TFLOP/s; replay {REPLAY_US:.3f} us")
    print("Kernel times below are raw (resource floor + one replay/call); speedups are the judge's replay-adjusted ratios.")
    for p in PROBLEMS:
        r = problems[p]
        print(f"\n== {p}")
        print(f"{'load':<13}{'spec raw ms':>13}{'4.25 raw ms':>14}{'spec x':>10}{'4.25 x':>10}{'stock ms':>12}")
        for load, x in r["loads"].items():
            print(f"{load:<13}{fmt_ms(x['raw_spec']):>13}{fmt_ms(x['raw_measured']):>14}"
                  f"{x['speedup_spec']:>10.3f}{x['speedup_measured']:>10.3f}{fmt_ms(x['stock']):>12}")
        g = r["gm"]
        print(f"{'geomean':<13}{fmt_ms(g['raw_spec']):>13}{fmt_ms(g['raw_measured']):>14}"
              f"{g['speedup_spec']:>10.3f}{g['speedup_measured']:>10.3f}{fmt_ms(g['stock']):>12}")
        print(f"practical model {r['practical_speedup']:.3f}x; audited NVFP4 issue binds: {r['issue_binds']}")

    print("\n== serving step (raw ms, perfect resource overlap)")
    print(f"{'c':<4}{'spec':>10}{'4.25 TB/s':>12}{'spec bottleneck':>22}{'4.25 bottleneck':>22}")
    for c in serving.RUNGS:
        a = serv["rates"]["spec"]["details"]["steps"][c]["floor"]
        b = serv["rates"]["measured"]["details"]["steps"][c]["floor"]
        print(f"{c:<4}{a.raw_us/1e3:>10.3f}{b.raw_us/1e3:>12.3f}{a.bottleneck:>22}{b.bottleneck:>22}")
    for name in ("spec", "measured"):
        d = serv["rates"][name]
        print(f"{name}: prefill conversation {d['details']['conversation_us']/1e6:.3f} s "
              f"(first {d['details']['first']['raw_us']/1e3:.1f} ms, later {d['details']['later']['raw_us']/1e3:.1f} ms each)")
        for tau, s in d["scores"].items():
            print(f"  tau {tau:.2f}: score {s['score_extrapolated']:.3f}x extrapolated / {s['score_capped']:.3f}x capped")

    print("\n== gap snapshot")
    for p, r in problems.items():
        ours, receipt = best_receipt(p)
        print(f"{p}: bound {r['gm']['speedup_spec']:.3f}x, practical {r['practical_speedup']:.3f}x, "
              f"ours {ours:.3f}x ({receipt}), leader {LEADERS[p]:.3f}x")
    print(f"serving: bound {serv['rates']['spec']['scores'][4.16]['score_capped']:.3f}x, "
          f"practical {serv['practical_score']:.3f}x, stock 1.000x, no receipts/leader (track closed)")

    print("\n== outside rules (not admissible; spec-rate capped score)")
    for name, x in serv["scenarios"].items():
        print(f"{name}: {x['score']:.3f}x score, {x['gain']:.3f}x over legal tau-4.16 bound, c32 step {x['step32_us']/1e3:.3f} ms")


if __name__ == "__main__":
    main()
