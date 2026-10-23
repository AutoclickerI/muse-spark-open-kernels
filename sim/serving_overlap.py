"""Price measured stream overlap, including prefill/decode readiness.

The aggregate ceiling assumes a full decode batch is always ready behind
another user's prefill. The event model removes that assumption: each user
must finish prefill before generating 220 tokens and submitting its next
turn. Both are models, not H200 serving measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from types import SimpleNamespace
from pathlib import Path

from sim import hw, muse, serving, serving_receipts


def experiment(path):
    data = json.load(open(path))
    samples = data["samples"]
    result = {}
    for priority in (-1, 0):
        rows = []
        for rep in sorted({row["rep"] for row in samples}):
            values = {(row["mode"], row["decode_priority"]): row
                      for row in samples if row["rep"] == rep}
            p = values["prefill", -1]["total_ms"]
            d = values["verify", -1]["total_ms"]
            c = values["concurrent", priority]
            pe, de = c["prefill_done_ms"], c["verify_done_ms"]
            common = min(pe, de)
            # Beyond the first completion, the remaining forward runs alone.
            prate = (p - max(0, pe - de)) / common
            drate = (d - max(0, de - pe)) / common
            rows.append(dict(rep=rep, eta=(p + d - c["total_ms"]) / d,
                             prefill_rate=prate, verify_rate=drate,
                             serial_consistency=values["serial", -1]["total_ms"] / (p + d)))
        result[priority] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("eta", "prefill_rate", "verify_rate", "serial_consistency")}
        result[priority]["samples"] = rows
    return result


def event_model(c, first_us, later_us, steps, *, eta=0, decode_rate=1,
                min_ready=1, overlap=False, turns=13, tau=4.16, first_chunks=3):
    # One slot/user, one sticky chunked request, FIFO new requests.
    # The first-turn benchmark contains three representative 16K chunks.
    users = [dict(turn=0, remaining=219.0) for _ in range(c)]
    queue = list(range(c))
    ready = set()
    prefill = None
    decode = None
    now = 0.0
    completed = 0
    hist = {}
    first_tokens = []
    while completed < c:
        if prefill is None and queue:
            uid = queue.pop(0)
            chunks = first_chunks if users[uid]["turn"] == 0 else 1
            duration = first_us / first_chunks if users[uid]["turn"] == 0 else later_us
            prefill = dict(uid=uid, chunks=chunks, duration=duration, left=duration)
        if decode is None and ready:
            if prefill is None or (overlap and len(ready) >= min_ready):
                cohort = tuple(sorted(ready))
                duration = serving._loglog(steps, len(cohort))
                decode = dict(cohort=cohort, left=duration)
                hist[len(cohort)] = hist.get(len(cohort), 0) + 1
        if prefill is None and decode is None:
            raise RuntimeError("event model made no progress")
        both = prefill is not None and decode is not None
        drate = decode_rate if both else 1.0
        # Each unit of isolated decode work adds (1-eta) units to prefill.
        prate = 1 - (1 - eta) * decode_rate if both else 1.0
        tp = prefill["left"] / prate if prefill is not None and prate > 0 else math.inf
        td = decode["left"] / drate if decode is not None else math.inf
        elapsed = min(tp, td)
        if not math.isfinite(elapsed):
            raise RuntimeError("zero progress rates")
        now += elapsed
        if prefill is not None:
            prefill["left"] -= elapsed * prate
        if decode is not None:
            decode["left"] -= elapsed * drate
        if prefill is not None and prefill["left"] < 1e-6:
            uid = prefill["uid"]
            prefill["chunks"] -= 1
            if prefill["chunks"]:
                prefill["left"] = prefill["duration"]
            else:
                ready.add(uid)
                users[uid]["remaining"] = 219.0
                if users[uid]["turn"] == 0:
                    first_tokens.append(now)
                prefill = None
        if decode is not None and decode["left"] < 1e-6:
            for uid in decode["cohort"]:
                users[uid]["remaining"] -= tau
                if users[uid]["remaining"] <= 0:
                    ready.remove(uid)
                    users[uid]["turn"] += 1
                    if users[uid]["turn"] < turns:
                        queue.append(uid)
                    else:
                        completed += 1
            decode = None
    return dict(window_us=now, decode_batch_histogram=hist,
                first_turn_last_ttft_us=first_tokens[c - 1])


def mixed_model(c, first_us, later_us, steps, *, min_ready=1,
                turns=13, tau=4.16, extra_us=0, first_chunks=3,
                first_extra_fraction=0, later_extra_fraction=0, measured_batch=32,
                mixed_tau=None):
    """One verify block per ready user per prefill chunk.

    extra_us=0 is an optimistic ceiling: even draft/head/accept work and the
    extra target rows are free. A new prefill user joins only the next chunk.
    """
    users = [dict(turn=0, remaining=219.0) for _ in range(c)]
    queue, ready = list(range(c)), set()
    active, now, finished = None, 0.0, 0
    hist = {"mixed": {}, "standalone": {}}

    def accept(cohort, amount=tau):
        nonlocal finished
        for uid in cohort:
            users[uid]["remaining"] -= amount
            if users[uid]["remaining"] <= 0:
                ready.remove(uid)
                users[uid]["turn"] += 1
                if users[uid]["turn"] == turns:
                    finished += 1
                else:
                    queue.append(uid)

    while finished < c:
        if active is None and queue:
            uid = queue.pop(0)
            chunks = first_chunks if users[uid]["turn"] == 0 else 1
            duration = first_us / first_chunks if users[uid]["turn"] == 0 else later_us
            active = [uid, chunks, duration]
        if active is not None:
            cohort = tuple(ready) if len(ready) >= min_ready else ()
            fraction = first_extra_fraction if users[active[0]]["turn"] == 0 else later_extra_fraction
            target_extra = active[2] * fraction * len(cohort) / measured_batch
            extra = (extra_us(len(cohort)) if callable(extra_us) else extra_us) if cohort else 0
            now += active[2] + target_extra + extra
            if cohort:
                hist["mixed"][len(cohort)] = hist["mixed"].get(len(cohort), 0) + 1
                accept(cohort, tau if mixed_tau is None else mixed_tau)
            active[1] -= 1
            if active[1] == 0:
                ready.add(active[0])
                users[active[0]]["remaining"] = 219.0
                active = None
        elif ready:
            cohort = tuple(ready)
            now += serving._loglog(steps, len(cohort))
            hist["standalone"][len(cohort)] = hist["standalone"].get(len(cohort), 0) + 1
            accept(cohort)
        else:
            raise RuntimeError("mixed model made no progress")
    return dict(window_us=now, batch_histogram=hist)


def price(eta_by_rung, decode_rate_by_rung, first_chunks=3, first_scale=1, mixed_cost=None):
    hw.EFF["allreduce_us"] = 5.5
    args = SimpleNamespace(turns=13, tau=4.16, host_us=0.0, interactivity="e2e")
    stock = serving.curve("stock", args)
    families = {}
    for name, pd in (("practical_decode_stock_prefill", "stock"),
                     ("practical_all", "practical")):
        families[name] = (
            serving.curve("practical", args, prefill_design=pd), stock,
            serving.prefill_load_us("first turn", pd),
            serving.prefill_load_us("turn 1", pd),
            serving.prefill_load_us("first turn", "stock"))
    decode = serving_receipts._benchmarks(serving_receipts.DECODE)
    prefill = serving_receipts._benchmarks(serving_receipts.PREFILL)
    accounted = serving_receipts.accounting_sensitivity(decode, prefill)
    first, later = [sum(r["case"]["weight"] * r["best_ns"] / 1e3
                       for r in prefill if r["case"]["load"] == load)
                    for load in ("first turn", "turn 1")]
    families["receipts_complete_accounting"] = (
        accounted["candidate"], accounted["stock"], first, later,
        sum(r["case"]["weight"] * r["baseline_ns"] / 1e3
            for r in prefill if r["case"]["load"] == "first turn"))

    def adjust_first(curve, first):
        changed = {}
        for c, row in curve.items():
            old_window = 13 * 220 * c * 1e6 / (8 * row["per_gpu"])
            delta = c * first * (first_scale - 1)
            factor = (old_window + delta) / old_window
            changed[c] = {**row, "per_gpu": row["per_gpu"] / factor,
                          "per_user": row["per_user"] / factor,
                          "prefill_share": (old_window * row["prefill_share"] + delta) / (old_window + delta)}
        return changed
    result = {}
    for name, (original, stock, first, later, stock_first) in families.items():
        original, stock = adjust_first(original, first), adjust_first(stock, stock_first)
        first *= first_scale
        steps = {c: row["step_us"] for c, row in original.items()}
        changed = {}
        modeled = {}
        readiness, mixed, priced_mixed = {}, {}, {}
        mixed_curve = {}
        generic_curve = {}
        priced_mixed_curve = {}
        for c, row in original.items():
            eta = eta_by_rung.get(c, 0)
            # Bound available work by another user's prefill time.
            saved_share = eta * min(1 - row["prefill_share"], row["prefill_share"] * (c - 1) / c)
            ratio = 1 - saved_share
            changed[c] = {**row, "per_gpu": row["per_gpu"] / ratio,
                          "per_user": row["per_user"] / ratio}
            baseline = event_model(c, first, later, steps, first_chunks=first_chunks)
            best = baseline
            threshold = c
            # Min-ready=c also represents the stock no-overlap schedule.
            for b in range(1, c):
                candidate = event_model(c, first, later, steps, eta=eta,
                                        decode_rate=decode_rate_by_rung.get(c, 1),
                                        min_ready=b, overlap=eta > 0, first_chunks=first_chunks)
                if candidate["window_us"] < best["window_us"]:
                    best, threshold = candidate, b
            readiness[c] = dict(stock=baseline, best=best, min_ready=threshold,
                                window_gain=baseline["window_us"] / best["window_us"] - 1)
            actual_ratio = best["window_us"] / baseline["window_us"]
            modeled[c] = {**row, "per_gpu": row["per_gpu"] / actual_ratio,
                          "per_user": row["per_user"] / actual_ratio}
            mixed_best = min(
                (mixed_model(c, first, later, steps, min_ready=b, first_chunks=first_chunks)
                 for b in range(1, c + 1)),
                key=lambda value: value["window_us"])
            mixed[c] = mixed_best
            mixed_ratio = min(1, mixed_best["window_us"] / baseline["window_us"])
            mixed_curve[c] = {**row, "per_gpu": row["per_gpu"] / mixed_ratio,
                              "per_user": row["per_user"] / mixed_ratio}
            generic = min((mixed_model(c, first, later, steps, min_ready=b,
                                      first_chunks=first_chunks, mixed_tau=1)
                           for b in range(1, c + 1)), key=lambda value: value["window_us"])
            generic_ratio = min(1, generic["window_us"] / baseline["window_us"])
            generic_curve[c] = {**row, "per_gpu": row["per_gpu"] / generic_ratio,
                                "per_user": row["per_user"] / generic_ratio}
            if mixed_cost:
                def draft_head_cost(b):
                    extras = serving.extras_us("practical", b, 0)
                    payload = b * serving.BLOCK * muse.HIDDEN * 2
                    draft_ar = 2 * muse.DRAFT["layers"] * hw.allreduce_us(payload)
                    return extras["drafter"] + extras["lm_head"] + draft_ar

                priced_best = min((mixed_model(
                    c, first, later, steps, min_ready=b, first_chunks=first_chunks,
                    first_extra_fraction=mixed_cost["first_fraction"],
                    later_extra_fraction=mixed_cost["later_fraction"],
                    measured_batch=mixed_cost["batch"], extra_us=draft_head_cost)
                    for b in range(1, c + 1)), key=lambda value: value["window_us"])
                priced_mixed[c] = priced_best
                priced_ratio = min(1, priced_best["window_us"] / baseline["window_us"])
                priced_mixed_curve[c] = {**row, "per_gpu": row["per_gpu"] / priced_ratio,
                                        "per_user": row["per_user"] / priced_ratio}
        before = serving.score(original, stock, capped=True)[0]
        aggregate = serving.score(changed, stock, capped=True)[0]
        feasible = serving.score(modeled, stock, capped=True)[0]
        mixed_score = serving.score(mixed_curve, stock, capped=True)[0]
        generic_score = serving.score(generic_curve, stock, capped=True)[0]
        priced_score = serving.score(priced_mixed_curve, stock, capped=True)[0] if mixed_cost else None
        result[name] = dict(before=before, aggregate_ceiling=aggregate,
                            aggregate_relative_gain=aggregate / before - 1,
                            ready_model_score=feasible,
                            ready_model_relative_gain=feasible / before - 1,
                            ideal_mixed_score=mixed_score,
                            ideal_mixed_relative_gain=mixed_score / before - 1,
                            generic_one_token_ceiling=generic_score / before - 1,
                            measured_mixed_score=priced_score,
                            measured_mixed_relative_gain=priced_score / before - 1 if priced_score else None,
                            priced_mixed=priced_mixed,
                            mixed=mixed,
                            readiness=readiness)
    return result


def check_model():
    steps = {1: 10.0, 4: 13.0, 16: 18.0, 32: 24.0}
    for c in serving.RUNGS:
        baseline = event_model(c, 100.0, 20.0, steps)
        expected = c * (100.0 + 12 * 20.0) + 13 * math.ceil(219 / 4.16) * steps[c]
        assert math.isclose(baseline["window_us"], expected, rel_tol=1e-10)
        if c == 1:
            candidate = event_model(c, 100.0, 20.0, steps, eta=1,
                                    decode_rate=1, overlap=True)
            assert candidate == baseline
        assert math.isclose(mixed_model(c, 100.0, 20.0, steps, min_ready=c)["window_us"],
                            baseline["window_us"], rel_tol=1e-10)
        for threshold in range(1, c):
            ideal = event_model(c, 100.0, 20.0, steps, eta=1,
                                min_ready=threshold, overlap=True)
            assert ideal["window_us"] >= c * (100.0 + 12 * 20.0) - 1e-6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measurement", action="append", default=[])
    ap.add_argument("--mixed-measurement", action="append", default=[])
    ap.add_argument("--eta", type=float, default=1)
    ap.add_argument("--decode-rate", type=float, default=1)
    ap.add_argument("--priority", type=int, choices=(-1, 0), default=-1)
    ap.add_argument("--first-chunks", type=int, default=3)
    ap.add_argument("--scale-first-prefill", action="store_true",
                    help="scale first-turn timing from three benchmark chunks to --first-chunks")
    ap.add_argument("--out")
    args = ap.parse_args()
    check_model()
    experiments = {path: experiment(path) for path in args.measurement}
    eta = {c: args.eta for c in (16, 32)}
    rates = {c: args.decode_rate for c in (16, 32)}
    for path, data in experiments.items():
        c = json.load(open(path))["args"]["concurrency"]
        eta[c] = max(0, min(1, data[args.priority]["eta"]))
        rates[c] = max(0.01, min(1, data[args.priority]["verify_rate"]))
    scale = args.first_chunks / 3 if args.scale_first_prefill else 1
    mixed_cost = None
    if args.mixed_measurement:
        measured = {}
        batches = set()
        for path in args.mixed_measurement:
            data = json.load(open(path))
            ratios = []
            for rep in sorted({r["rep"] for r in data["samples"]}):
                times = {r["mode"]: r["ms"] for r in data["samples"] if r["rep"] == rep}
                ratios.append(times["mixed"] / times["prefill"] - 1)
            measured[data["args"]["prefill_length"]] = max(0, statistics.median(ratios))
            batches.add(data["args"]["concurrency"])
        if len(batches) != 1:
            raise ValueError("mixed measurements must use the same concurrency")
        mixed_cost = dict(first_fraction=measured[16384], later_fraction=measured[753], batch=batches.pop())
    result = dict(measurements=experiments, applied_eta=eta,
                  applied_decode_rate=rates, first_chunks=args.first_chunks,
                  first_prefill_scale=scale,
                  mixed_cost=mixed_cost,
                  price=price(eta, rates, args.first_chunks, scale, mixed_cost))
    root = Path(__file__).resolve().parents[1]
    result["inputs"] = {
        "sources_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__), root / "sim/serving.py",
                                     root / "sim/hw.py", root / "sim/serving_receipts.py",
                                     serving_receipts.DECODE, serving_receipts.PREFILL)},
        "stock_calibration": {name: serving._stock_shapes(name)
                              for name in ("muse_decode_layer", "muse_prefill_layer")},
        "allreduce_model": dict(hw.EFF),
    }
    out = json.dumps(result, indent=2) + "\n"
    if args.out:
        open(args.out, "w").write(out)
    print(out)


if __name__ == "__main__":
    main()
