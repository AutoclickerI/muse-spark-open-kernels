"""Compose the current H200 receipt set through the serving score model.

This keeps the receipt composition separate from ``sim.serving``'s generic
bound/practical designs: the decode-layer and prefill receipts provide the
measured stock and candidate times, while ``sim.serving`` supplies the
conversation and score calculation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from sim import bound, hw, muse, serving
from sim.problems import layer_routing

ROOT = Path(__file__).resolve().parent.parent
DECODE = ROOT / "problems/muse_decode_layer/profiles/cand_r3_v9.0e94268cf5654012.json"
PREFILL = ROOT / "problems/muse_prefill_layer/profiles/cand_score_portfolio_routing.a30fb411b0c74ffb.json"


def _benchmarks(path: Path) -> list[dict]:
    return json.loads(path.read_text())["benchmarks"]


def _loglog(points: dict[int, float], x: int) -> float:
    xs = sorted(points)
    lo, hi = (
        (xs[0], xs[1])
        if x <= xs[0]
        else (xs[-2], xs[-1])
        if x >= xs[-1]
        else next((a, b) for a, b in zip(xs, xs[1:]) if a <= x <= b)
    )
    w = (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return math.exp(math.log(points[lo]) * (1 - w) + math.log(points[hi]) * w)


def step_breakdown(decode: dict) -> dict:
    """Attribute the published receipt composition; do not hide unpriced work.

    Kernel profile durations can overlap and include PDL waits. Normalize their
    shares to each receipt's best wall time; this is attribution, not isolated
    kernel latency. Interpolate each layer's wall time and renormalize component
    shares at c=4, where no receipt exists.
    """
    profiles = {}
    for row, profile in zip(decode["benchmarks"], decode["profiles"]):
        assert row["case"] == profile["case"]
        if not row["case"]["load"].startswith("verify"):
            continue
        report = profile["report"]
        total = float(re.search(r"total\s+([0-9.]+) µs", report).group(1))
        components = dict(moe=0.0, attention=0.0, projections=0.0)
        for line in report.splitlines():
            match = re.match(r"(.+?)\s+([0-9.]+) µs\s+", line)
            if not match:
                continue
            name, us = match.group(1).strip(), float(match.group(2))
            if "moe_fused" in name:
                components["moe"] += us
            elif any(
                part in name
                for part in ("flash::", "_qk_prep_kernel", "prepare_varlen")
            ):
                components["attention"] += us
            elif any(part in name for part in ("_gemv_kernel", "nvjet_", "cublasLt")):
                components["projections"] += us
        components["glue"] = total - sum(components.values())
        assert components["glue"] >= 0
        wall = row["best_ns"] / 1e3
        profiles[(row["case"]["batch"], row["case"]["layer"])] = (
            wall,
            {key: wall * value / total for key, value in components.items()},
        )
    result = {}
    for c in serving.RUNGS:
        attributed = dict(moe=0.0, attention=0.0, projections=0.0, glue=0.0)
        for layer, count in (("sliding", 46), ("full", 16)):
            wall = _loglog({b: profiles[b, layer][0] for b in (1, 16, 32)}, c)
            parts = {
                key: _loglog({b: profiles[b, layer][1][key] for b in (1, 16, 32)}, c)
                for key in attributed
            }
            scale = wall / sum(parts.values())
            for key in attributed:
                attributed[key] += count * parts[key] * scale
        extras = serving.extras_us("practical", c, 0.0)
        ar = hw.allreduce_us(c * serving.BLOCK * muse.HIDDEN * 2)
        rows = {
            **attributed,
            "attention_output_ar": muse.LAYERS * ar,
            "expert_ar": muse.LAYERS * ar,
            "pooler_ag": None,
            "router_ag": None,
            "unpooler_ag": None,
            "drafter": extras["drafter"],
            "drafter_ar": 2 * muse.DRAFT["layers"] * ar,
            "lm_head": extras["lm_head"],
            "sampling_verify": None,
            "graph_replay": None,
            "cpu_scheduler": None,
        }
        total = sum(value for value in rows.values() if value is not None)
        payload = c * serving.BLOCK * muse.HIDDEN * 2
        ar_floor = muse.LAYERS * (
            bound.allreduce_us(payload) + bound.allreduce_us(payload / 2)
        )
        draft_work, _ = bound.drafter_step(c)
        draft_floor = bound.floor(draft_work, bound.SPEC_BW).adjusted_us
        head_floor = bound.floor(bound.lm_head_step(c), bound.SPEC_BW).adjusted_us
        result[c] = {
            "components_us": rows,
            "priced_total_us": total,
            "recoverable_us": {
                "target_ar": rows["attention_output_ar"] + rows["expert_ar"] - ar_floor,
                "drafter_kernels": max(0.0, rows["drafter"] - draft_floor),
                "lm_head": max(0.0, rows["lm_head"] - head_floor),
            },
            "omitted_gather_estimates_us": {
                # Same latency/bandwidth proxy as serving's sharded-dense model.
                "pooler": muse.LAYERS * hw.allreduce_us(payload / 2)
                if c <= 16
                else 0.0,
                "router": 0.0,  # remains replicated
                "unpooler": muse.LAYERS * hw.allreduce_us(payload) if c <= 16 else 0.0,
            },
        }
    return result


def accounting_sensitivity(decode: list[dict], prefill: list[dict]) -> dict:
    """Price known omissions without claiming a new measured serving result.

    Use candidate medians instead of best samples, correct expert AR width,
    add the two BF16-expert layer premium and enabled projection gathers,
    and model the already-ported FC sharding. Keep the received prefill model
    unchanged. The two full-model graph replays are an explicit floor proxy;
    host/sampling and the true H200 custom-AR/gather overlap remain uncalibrated.
    """
    curves = {}
    for candidate in (False, True):
        design = "practical" if candidate else "stock"
        table = {
            (r["case"]["batch"], r["case"]["layer"]): r[
                "median_ns" if candidate else "baseline_ns"
            ]
            / 1e3
            for r in decode
            if r["case"]["load"].startswith("verify")
        }
        pf = sum(
            r["case"]["weight"]
            * (1 if r["case"]["load"] == "first turn" else 12)
            * r["best_ns" if candidate else "baseline_ns"]
            / 1e3
            for r in prefill
            if r["case"]["load"] in ("first turn", "turn 1")
        )
        curve = {}
        for c in serving.RUNGS:
            tokens = c * serving.BLOCK
            layers = sum(
                count * _loglog({b: table[b, layer] for b in (1, 16, 32)}, c)
                for layer, count in (("sliding", 46), ("full", 16))
            )
            touched, _ = layer_routing(tokens)
            bf16_premium = serving.BF16_EXPERT_LAYERS * serving._gemm_us(
                design, touched * (serving.BF16_EXPERT_BYTES - muse.EXPERT_BYTES), 0
            )
            extra = serving.extras_us(design, c, 0.0)
            payload = tokens * muse.HIDDEN * 2
            ar = muse.LAYERS * (hw.allreduce_us(payload) + hw.allreduce_us(payload / 2))
            ar += 2 * muse.DRAFT["layers"] * hw.allreduce_us(payload)
            fc_save = (
                hw.stream_us(
                    muse.DRAFT["context_layers"]
                    * muse.DRAFT["hidden"] ** 2
                    * 2
                    * (1 - 1 / muse.TP)
                )
                if candidate
                else 0.0
            )
            gathers = hw.allreduce_us(payload) if candidate else 0.0  # sharded draft FC
            if candidate and c <= 16:
                gathers += muse.LAYERS * (
                    hw.allreduce_us(payload / 2) + hw.allreduce_us(payload)
                )
            elif not candidate and c == 1:
                gathers += muse.LAYERS * hw.allreduce_us(payload)
            replay = 2 * bound.REPLAY_US
            step = (
                layers
                + bf16_premium
                + extra["drafter"]
                - fc_save
                + extra["lm_head"]
                + ar
                + gathers
                + replay
            )
            window = c * pf + 13 * serving.OUTPUT_TOKENS / serving.DEFAULT_TAU * step
            curve[c] = {
                "step_us": step,
                "per_gpu": 13 * serving.OUTPUT_TOKENS * c / window * 1e6 / 8,
                "per_user": 13 * serving.OUTPUT_TOKENS / window * 1e6,
                "prefill_share": c * pf / window,
            }
        curves["candidate" if candidate else "stock"] = curve
    score, _ = serving.score(curves["candidate"], curves["stock"], capped=True)
    return {**curves, "score_capped": score}


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
        prefill_stock[load] = (
            prefill_stock.get(load, 0.0) + weight * row["baseline_ns"] / 1e3
        )
        prefill_candidate[load] = (
            prefill_candidate.get(load, 0.0) + weight * row["best_ns"] / 1e3
        )

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
            step_us = layer_stack(c, candidate) + sum(
                serving.extras_us(design, c, 0.0).values()
            )
            window = c * prefill_us + 13 * serving.OUTPUT_TOKENS / serving.DEFAULT_TAU * step_us
            out[c] = {
                "step_us": step_us,
                "per_gpu": 13 * serving.OUTPUT_TOKENS * c / window * 1e6 / 8,
                "per_user": 13 * serving.OUTPUT_TOKENS / window * 1e6,
                "prefill_share": c * prefill_us / window,
            }
        return out

    hw.EFF["allreduce_us"] = 5.5
    breakdown = step_breakdown(json.loads(DECODE.read_text()))
    stock = curve(False)
    candidate = curve(True)
    score, per_rung = serving.score(candidate, stock)
    capped, capped_per_rung = serving.score(candidate, stock, capped=True)
    recovery_scores = {}
    for component in ("target_ar", "drafter_kernels", "lm_head"):
        recovered = {}
        for c, row in candidate.items():
            saved = breakdown[c]["recoverable_us"][component]
            window_ratio = 1 - saved / row["step_us"] * (1 - row["prefill_share"])
            recovered[c] = {
                **row,
                "per_user": row["per_user"] / window_ratio,
                "per_gpu": row["per_gpu"] / window_ratio,
            }
        recovery_scores[component] = serving.score(recovered, stock, capped=True)[0]
    print(
        json.dumps(
            {
                "receipts": {
                    "decode": str(DECODE.relative_to(ROOT)),
                    "prefill": str(PREFILL.relative_to(ROOT)),
                },
                "stock": stock,
                "candidate": candidate,
                "score_extrapolated": score,
                "score_capped": capped,
                "per_rung": per_rung,
                "per_rung_capped": capped_per_rung,
                "candidate_breakdown": breakdown,
                "unowned_recovery_score_capped": recovery_scores,
                "accounting_sensitivity": accounting_sensitivity(decode, prefill),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
