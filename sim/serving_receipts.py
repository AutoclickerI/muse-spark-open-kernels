"""Compose the current H200 receipt set through the serving score model.

This keeps the receipt composition separate from ``sim.serving``'s generic
bound/practical designs: the decode-layer and prefill receipts provide the
measured stock and candidate times, while ``sim.serving`` supplies the
conversation and score calculation.
"""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path

from sim import hw, muse, serving

ROOT = Path(__file__).resolve().parent.parent
DECODE = ROOT / "problems/muse_decode_layer/profiles/cand_moe7.d2bf1ced702948a4.json"
PREFILL = ROOT / "problems/muse_prefill_layer/profiles/cand_v11.6a9cd943fe404d74.json"


def _benchmarks(path: Path) -> list[dict]:
    return json.loads(path.read_text())["benchmarks"]


def _loglog(points: dict[int, float], x: int) -> float:
    xs = sorted(points)
    lo, hi = (xs[0], xs[1]) if x <= xs[0] else (xs[-2], xs[-1]) if x >= xs[-1] else next(
        (a, b) for a, b in zip(xs, xs[1:]) if a <= x <= b
    )
    w = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(points[lo]) * (1 - w) + math.log(points[hi]) * w)


def main() -> None:
    decode = _benchmarks(DECODE)
    prefill = _benchmarks(PREFILL)
    stock_layer: dict[tuple[int, str], float] = {}
    candidate_layer: dict[tuple[int, str], float] = {}
    for row in decode:
        case = row["case"]
        key = (case["batch"], case["layer"])
        stock_layer[key] = row["baseline_ns"] / 1e3
        candidate_layer[key] = row["best_ns"] / 1e3

    prefill_stock: dict[str, float] = {}
    prefill_candidate: dict[str, float] = {}
    for row in prefill:
        load = row["case"]["load"]
        weight = row["case"]["weight"]
        prefill_stock[load] = prefill_stock.get(load, 0.0) + weight * row["baseline_ns"] / 1e3
        prefill_candidate[load] = prefill_candidate.get(load, 0.0) + weight * row["best_ns"] / 1e3

    def layer_stack(c: int, candidate: bool) -> float:
        table = candidate_layer if candidate else stock_layer
        by_layer = {
            layer: {batch: table[(batch, layer)] for batch in (1, 16, 32)}
            for layer in ("sliding", "full")
        }
        return 46 * _loglog(by_layer["sliding"], c) + 16 * _loglog(by_layer["full"], c)

    def prefill_time(candidate: bool) -> float:
        table = prefill_candidate if candidate else prefill_stock
        return table["first turn"] + 12 * table["turn 1"]

    def curve(candidate: bool) -> dict[int, dict[str, float]]:
        design = "practical" if candidate else "stock"
        prefill_us = prefill_time(candidate)
        out = {}
        for c in serving.RUNGS:
            step_us = layer_stack(c, candidate) + sum(serving.extras_us(design, c, 0.0).values())
            window = c * prefill_us + 13 * serving.OUTPUT_TOKENS / 4.16 * step_us
            out[c] = {
                "step_us": step_us,
                "per_gpu": 13 * serving.OUTPUT_TOKENS * c / window * 1e6 / 8,
                "per_user": 13 * serving.OUTPUT_TOKENS / window * 1e6,
                "prefill_share": c * prefill_us / window,
            }
        return out

    hw.EFF["allreduce_us"] = 5.5
    stock = curve(False)
    candidate = curve(True)
    score, per_rung = serving.score(candidate, stock)
    capped, capped_per_rung = serving.score(candidate, stock, capped=True)
    print(json.dumps({
        "receipts": {"decode": str(DECODE.relative_to(ROOT)), "prefill": str(PREFILL.relative_to(ROOT))},
        "stock": stock,
        "candidate": candidate,
        "score_extrapolated": score,
        "score_capped": capped,
        "per_rung": per_rung,
        "per_rung_capped": capped_per_rung,
    }, indent=2))


if __name__ == "__main__":
    main()
