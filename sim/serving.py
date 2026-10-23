"""Serving model: one step of the node (8xH200; TP8 attention and dense, EP8 experts) under the judge's
recipe (DFlash, each request verifying an 8-token block per step), the throughput/interactivity
curve over the judge's rungs, and the score: our throughput at stock's interactivity over stock's,
averaged over rungs (the site's "Score" card; how it handles points off our curve isn't published).

All times are per GPU per step; the 8 GPUs run the same step and the busiest sets the pace.
Stock's layer and prefill times are the judge's own SGLang measurements (sim.calibrate's runs);
the rest of stock's step (LM head, drafter, all-reduces) is modeled like ours at cuBLAS rates.

    python -m sim.serving [--tau 4] [--allreduce-us 8] [--cpu-us 0]
"""

import argparse
import math

from sim import hw, muse
from sim.calibrate import latest_runs, shapes
from sim.problems import shape_bounds

RUNGS = (1, 4, 16, 32)
BLOCK = 8              # --speculative-num-draft-tokens 8
OUTPUT_TOKENS = 220    # every request returns exactly 220 tokens
TURN_TOKENS = 753      # new tokens per agentic turn (muse_prefill_layer's turn loads)
CTX = 80_000
STOCK_GEMM = (2.8, 3.54e12)   # SGLang's cuBLAS on H200: fixed us, bytes/s (sim.calibrate, bf16_proj)


def _loglog(points: dict[int, float], x: float) -> float:
    """Interpolate y(x) from {x: y} in log-log space (extrapolating from the end segments)."""
    xs = sorted(points)
    lo, hi = (xs[0], xs[1]) if x <= xs[0] else (xs[-2], xs[-1]) if x >= xs[-1] else \
        next((a, b) for a, b in zip(xs, xs[1:]) if a <= x <= b)
    w = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(points[lo]) * (1 - w) + math.log(points[hi]) * w)


def stock_layer_us(problem: str, select) -> dict[str, dict[int, float]]:
    """SGLang's measured per-layer time {layer: {batch: us}} for the shapes `select` keeps."""
    run = latest_runs().get(problem)
    if run is None:
        raise SystemExit(f"no calibration run for {problem} in runs/ (tools/okc_run.py check ...)")
    out: dict[str, dict[int, float]] = {}
    for r in shapes(run):
        c = r["case"]
        if select(c):
            out.setdefault(c["layer"], {})[c["batch"]] = r["sglang_us"]
    return out


def layer_stack_us(c: int, design: str) -> float:
    """The 62 target layers at batch c, q_len 8, ctx CTX."""
    if design == "stock":
        per = stock_layer_us("muse_decode_layer", lambda case: case.get("q_len") == BLOCK)
        return muse.SLIDING_LAYERS * _loglog(per["sliding"], c) + muse.FULL_LAYERS * _loglog(per["full"], c)
    kernels = {"practical": 8, "megakernel": 1, "expected": 1}[design]
    key = "expected" if design == "expected" else "practical"
    total = 0.0
    for layer, n in (("sliding", muse.SLIDING_LAYERS), ("full", muse.FULL_LAYERS)):
        b = {"batch": c, "q_len": BLOCK, "ctx": CTX, "layer": layer, "spread": "agentic" if c > 1 else "fixed"}
        total += n * shape_bounds("muse_decode_layer", b, kernels)[key]
    return total


def prefill_us(design: str) -> float:
    """One request's turn: TURN_TOKENS new tokens on CTX through the 62 layers."""
    if design == "stock":
        per = stock_layer_us("muse_prefill_layer", lambda case: case.get("q_len") == TURN_TOKENS and case["batch"] == 1)
        return muse.SLIDING_LAYERS * per["sliding"][1] + muse.FULL_LAYERS * per["full"][1]
    kernels, key = (10, "practical") if design != "expected" else (1, "expected")
    return sum(n * shape_bounds("muse_prefill_layer", {"batch": 1, "q_len": TURN_TOKENS, "ctx": CTX, "layer": layer},
                                kernels)[key]
               for layer, n in (("sliding", muse.SLIDING_LAYERS), ("full", muse.FULL_LAYERS)))


def extras_us(design: str, allreduce_us: float, cpu_us: float) -> dict[str, float]:
    """The step's work outside the 62 layers: LM head, drafter, all-reduces, host."""
    d = muse.DRAFT
    drafter_kernels = d["layers"] * 6 + 2        # qkv, attention, o, gate_up, down, norms; fc; head
    if design == "stock":
        fixed, bps = STOCK_GEMM
        gemm = lambda nbytes, kernels: kernels * fixed + nbytes / bps * 1e6   # noqa: E731
    else:
        bw = 1.0 if design == "expected" else hw.EFF["bw"]
        gemm = lambda nbytes, kernels: (0 if design == "expected" else hw.kernels_us(kernels)) + hw.stream_us(nbytes, bw)  # noqa: E731
    # Two all-reduces per target layer (after o and after the unpooler) and per drafter layer.
    reduces = 2 * (muse.LAYERS + d["layers"])
    return {"lm_head": gemm(muse.LM_HEAD_BYTES, 1), "drafter": gemm(muse.drafter_bytes(), drafter_kernels),
            "allreduce": reduces * (0 if design == "expected" else allreduce_us), "host": cpu_us}


def curve(design: str, tau: float, allreduce_us: float, cpu_us: float) -> dict[int, dict[str, float]]:
    """Per rung: step time, and (serving each user's turn prefill in line with the decode steps)
    throughput in tokens/s per GPU and interactivity in tokens/s per user."""
    extras = sum(extras_us(design, allreduce_us, cpu_us).values())
    prefill = prefill_us(design)
    out = {}
    for c in RUNGS:
        step = layer_stack_us(c, design) + extras
        round_us = c * prefill + OUTPUT_TOKENS / tau * step   # every user finishes one turn
        out[c] = {"step_us": step, "per_gpu": OUTPUT_TOKENS * c / round_us * 1e6 / 8,
                  "per_user": OUTPUT_TOKENS / round_us * 1e6, "prefill_share": c * prefill / round_us}
    return out


def score(ours: dict, stock: dict) -> tuple[float, dict[int, float]]:
    """Mean over rungs of our throughput at stock's interactivity over stock's throughput."""
    pts = {v["per_user"]: v["per_gpu"] for v in ours.values()}
    per = {}
    for c, s in stock.items():
        per[c] = _loglog(pts, s["per_user"]) / s["per_gpu"]
    return sum(per.values()) / len(per), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=4.0, help="tokens accepted per verify step (stock's DFlash)")
    ap.add_argument("--allreduce-us", type=float, default=hw.EFF["allreduce_us"])
    ap.add_argument("--cpu-us", type=float, default=0.0, help="host time per step not hidden by overlap")
    args = ap.parse_args()
    curves = {d: curve(d, args.tau, args.allreduce_us, args.cpu_us) for d in ("stock", "practical", "megakernel", "expected")}
    print(f"tau {args.tau} accepted tokens/step, all-reduce {args.allreduce_us} us, host {args.cpu_us} us/step, ctx {CTX:,}")
    for d, cv in curves.items():
        ex = extras_us(d, args.allreduce_us, args.cpu_us)
        print(f"\n== {d}: prefill {prefill_us(d) / 1e3:.1f} ms/turn; per step: "
              + ", ".join(f"{k} {v:.0f} us" for k, v in ex.items()))
        for c, v in cv.items():
            print(f"   c={c:<3} step {v['step_us'] / 1e3:6.2f} ms (layers {layer_stack_us(c, d) / 1e3:6.2f})"
                  f"  {v['per_gpu']:8.1f} tok/s/GPU  {v['per_user']:7.1f} tok/s/user  prefill {v['prefill_share']:.0%}")
        if d != "stock":
            mean, per = score(cv, curves["stock"])
            print(f"   score vs stock {mean:.2f}x  (" + ", ".join(f"c{c} {x:.2f}" for c, x in per.items()) + ")")


if __name__ == "__main__":
    main()
