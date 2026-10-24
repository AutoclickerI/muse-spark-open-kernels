"""Serving model: one step of the node (8xH200; TP8 attention and dense, EP8 experts) under the judge's
recipe (DFlash, each request verifying an 8-token block per step), whole multi-turn conversations,
the throughput/interactivity curve over the judge's rungs, and the score: our throughput at stock's
interactivity over stock's, averaged over rungs (the site's "Score" card; how points off our curve
are handled isn't published).

All times are per GPU; the 8 GPUs run the same step and the busiest sets the pace. Stock's layer and
prefill times are the judge's own SGLang measurements (sim.calibrate's runs); the rest of stock's
step (LM head, drafter, all-reduces, the two BF16-expert layers) is modeled at cuBLAS rates.

The workload is modeled on LMSYS's OpenHands simulation for Muse (day-0 blog, 2026-10-15): a
conversation is ~13 turns, a ~74K-token first turn, then ~753 new tokens per turn, 220 output tokens
per turn. The judge's conversations are hidden; prefill is served in line with decode (no overlap).

    python -m sim.serving [--tau 4.05] [--turns 13] [--allreduce-us 5.5] [--host-us 0] [--interactivity e2e|itl]
"""

import argparse
import math

from sim import hw, muse
from sim.calibrate import sglang_mean
from sim.problems import layer_routing, shape_bounds, spec, task_floors

RUNGS = (1, 4, 16, 32)
BLOCK = 8              # --speculative-num-draft-tokens 8
DEFAULT_TAU = 4.05  # InferenceX DFlash2, seven drafts plus the current token.
OUTPUT_TOKENS = 220    # every request returns exactly 220 tokens
CTX = 80_000
BF16_EXPERT_LAYERS = 2   # layers 0 and 61 keep BF16 experts (srt/layers/muse_spark_bf16_moe.py)
BF16_EXPERT_BYTES = 3 * muse.D_MOE * muse.D_MOE * 2
STOCK_GEMM = (2.8, 3.54e12)   # SGLang's cuBLAS on H200: fixed us, bytes/s (sim.calibrate, bf16_proj)


def _loglog(points: dict[float, float], x: float) -> float:
    """Interpolate y(x) from {x: y} in log-log space (extrapolating from the end segments)."""
    xs = sorted(points)
    lo, hi = (xs[0], xs[1]) if x <= xs[0] else (xs[-2], xs[-1]) if x >= xs[-1] else \
        next((a, b) for a, b in zip(xs, xs[1:]) if a <= x <= b)
    w = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(points[lo]) * (1 - w) + math.log(points[hi]) * w)


def _stock_shapes(problem: str) -> list[dict]:
    rows = sglang_mean(problem)
    if not rows:
        raise SystemExit(f"no calibration run for {problem} in runs/ (tools/okc_run.py send ...)")
    return rows


def _gemm_us(design: str, nbytes: float, kernels: int) -> float:
    if design == "stock":
        fixed, bps = STOCK_GEMM
        return kernels * fixed + nbytes / bps * 1e6
    if design == "expected":
        return hw.stream_us(nbytes, 1.0)
    return hw.kernels_us(kernels) + hw.stream_us(nbytes)


def layer_stack_us(c: int, design: str, shard_dense: bool = False) -> float:
    """The 62 target layers at batch c, q_len 8, ctx CTX (two of them with BF16 experts)."""
    tokens = c * BLOCK
    touched, _ = layer_routing(tokens)
    bf16_extra = BF16_EXPERT_LAYERS * _gemm_us(design, touched * (BF16_EXPERT_BYTES - muse.EXPERT_BYTES), 0)
    if design == "stock":
        per: dict[str, dict[float, float]] = {}
        for r in _stock_shapes("muse_decode_layer"):
            if r["case"].get("q_len") == BLOCK:
                per.setdefault(r["case"]["layer"], {})[r["case"]["batch"]] = r["sglang_us"]
        return muse.SLIDING_LAYERS * _loglog(per["sliding"], c) + muse.FULL_LAYERS * _loglog(per["full"], c) + bf16_extra
    kernels = {"practical": 8, "megakernel": 1, "expected": 1}[design]
    key = "expected" if design == "expected" else "practical"
    total = bf16_extra
    for layer, n in (("sliding", muse.SLIDING_LAYERS), ("full", muse.FULL_LAYERS)):
        b = {"batch": c, "q_len": BLOCK, "ctx": CTX, "layer": layer, "spread": "agentic" if c > 1 else "fixed"}
        total += n * shape_bounds("muse_decode_layer", b, kernels)[key]
    if shard_dense:  # each GPU reads 1/8 of the replicated pooler (and unpooler above 8 tokens), then
        # all-gathers the output (gemm_ag; priced like an all-reduce of the gathered output)
        saved = muse.POOLER_BYTES
        gathers = [tokens * muse.D_MOE * 2]
        if tokens > muse.SLICED_MAX_TOKENS:
            saved += muse.unpooler_bytes(tokens)
            gathers.append(tokens * muse.HIDDEN * 2)
        total -= muse.LAYERS * _gemm_us(design, saved * (1 - 1 / muse.TP), 0)
        if design != "expected":
            total += muse.LAYERS * sum(hw.allreduce_us(n) for n in gathers)
    return total


def prefill_load_us(load: str, design: str) -> float:
    """One of muse_prefill_layer's loads through the whole model ('turn 1': one request's ~753-token
    turn at 80K; 'first turn': a 64K-token first turn in 16K chunks)."""
    benches = spec("muse_prefill_layer")["benchmarks"]
    if design == "stock":
        times = {tuple(sorted((k, v) for k, v in r["case"].items() if k not in ("seed",))): r["sglang_us"]
                 for r in _stock_shapes("muse_prefill_layer")}
        return sum(b["weight"] * times[tuple(sorted((k, v) for k, v in b.items() if k != "seed"))]
                   for b in benches if b["load"] == load)
    kernels, key = (10, "practical") if design != "expected" else (1, "expected")
    return sum(b["weight"] * shape_bounds("muse_prefill_layer", b, kernels, tf)[key]
               for b, tf in zip(benches, task_floors("muse_prefill_layer")) if b["load"] == load)


def extras_us(design: str, c: int, host_us: float) -> dict[str, float]:
    """The step's work outside the 62 layers at batch c: LM head, drafter, all-reduces, host."""
    d = muse.DRAFT
    drafter_kernels = d["layers"] * 6 + 2        # qkv, attention, o, gate_up, down, norms; fc; head
    # After o and after the unpooler (target), after o and the MLP (drafter); [tokens, 8192] bf16.
    reduces = 2 * (muse.LAYERS + d["layers"])
    payload = c * BLOCK * muse.HIDDEN * 2
    return {"lm_head": _gemm_us(design, muse.LM_HEAD_BYTES, 1),
            "drafter": _gemm_us(design, muse.drafter_bytes(), drafter_kernels),
            "allreduce": 0.0 if design == "expected" else reduces * hw.allreduce_us(payload), "host": host_us}


def curve(design: str, args, shard_dense: bool = False, prefill_design: str | None = None) -> dict[int, dict[str, float]]:
    """Per rung: step time, and with every user running one whole conversation (its prefills
    in line with the decode steps), tokens/s per GPU and tokens/s per user. `prefill_design`
    prices the prefills differently from the decode steps (default: the same design).

    The site doesn't define interactivity beyond "tokens/s per user". args.interactivity 'e2e' counts
    the user's whole conversation time; 'itl' leaves out the user's own prefills (time to first token),
    as LMSYS's 1000 / inter-token latency does, while other users' prefills still stall its decode.
    Throughput's ratio to stock at a rung doesn't depend on whether input tokens are counted."""
    pd = prefill_design or design
    prefill = prefill_load_us("first turn", pd) + (args.turns - 1) * prefill_load_us("turn 1", pd)
    out_tokens = args.turns * OUTPUT_TOKENS
    out = {}
    for c in RUNGS:
        step = layer_stack_us(c, design, shard_dense) + sum(extras_us(design, c, args.host_us).values())
        window = c * prefill + out_tokens / args.tau * step
        user_us = window - prefill if args.interactivity == "itl" else window
        out[c] = {"step_us": step, "per_gpu": out_tokens * c / window * 1e6 / 8,
                  "per_user": out_tokens / user_us * 1e6, "prefill_share": c * prefill / window}
    return out


def score(ours: dict, stock: dict, capped: bool = False) -> tuple[float, dict[int, float]]:
    """Mean over rungs of our throughput at stock's interactivity over stock's throughput. Where
    stock's interactivity is outside our curve, the default extrapolates our curve; `capped` takes
    our nearest rung instead (we can't run past concurrency 32: --max-running-requests 32)."""
    pts = {v["per_user"]: v["per_gpu"] for v in ours.values()}
    lo, hi = min(pts), max(pts)
    per = {c: _loglog(pts, min(max(s["per_user"], lo), hi) if capped else s["per_user"]) / s["per_gpu"]
           for c, s in stock.items()}
    return sum(per.values()) / len(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="tokens per verify step (InferenceX DFlash2, seven drafts plus current token)")
    ap.add_argument("--turns", type=int, default=13, help="turns per conversation (LMSYS's OpenHands simulation)")
    ap.add_argument("--allreduce-us", type=float, default=hw.EFF["allreduce_us"], help="all-reduce latency (sim.hw)")
    ap.add_argument("--host-us", type=float, default=0.0, help="host time per step not hidden by overlap")
    ap.add_argument("--interactivity", choices=("e2e", "itl"), default="e2e",
                    help="tokens/s per user over the whole conversation, or without the user's own prefills")
    args = ap.parse_args()
    hw.EFF["allreduce_us"] = args.allreduce_us
    print(f"tau {args.tau}, {args.turns} turns/conversation, all-reduce {args.allreduce_us} us, host {args.host_us} us/step, "
          f"ctx {CTX:,}, interactivity {args.interactivity}")
    # name: (decode-step design, sharded pooler, prefill design)
    designs = {"stock": ("stock", False, "stock"), "practical": ("practical", False, "practical"),
               "practical decode, stock prefill": ("practical", False, "stock"),
               "stock decode, practical prefill": ("stock", False, "practical"),
               "practical + sharded pooler": ("practical", True, "practical"),
               "megakernel": ("megakernel", False, "megakernel"),
               "expected (4.8 TB/s, no overheads)": ("expected", False, "expected")}
    curves = {}
    for name, (d, shard, pd) in designs.items():
        curves[name] = cv = curve(d, args, shard, pd)
        ex = extras_us(d, 32, args.host_us)
        print(f"\n== {name}: prefill per conversation {(prefill_load_us('first turn', pd) + (args.turns - 1) * prefill_load_us('turn 1', pd)) / 1e3:,.0f} ms"
              f" (first turn {prefill_load_us('first turn', pd) / 1e3:,.0f}, later turns {prefill_load_us('turn 1', pd) / 1e3:.0f} each); per step at c=32: "
              + ", ".join(f"{k} {v:.0f} us" for k, v in ex.items()))
        for c, v in cv.items():
            print(f"   c={c:<3} step {v['step_us'] / 1e3:6.2f} ms (layers {layer_stack_us(c, d, shard) / 1e3:6.2f})"
                  f"  {v['per_gpu']:8.1f} tok/s/GPU  {v['per_user']:7.1f} tok/s/user  prefill {v['prefill_share']:.0%} of time")
        if name != "stock":
            for label, capped in (("extrapolated", False), ("capped", True)):
                mean, per = score(cv, curves["stock"], capped)
                print(f"   score vs stock, {label:12} {mean:.2f}x  (" + ", ".join(f"c{c} {x:.2f}" for c, x in per.items()) + ")")


if __name__ == "__main__":
    main()
