#!/usr/bin/env python3
"""Compare teacher-forced prompt and generated-token logprobs in two receipts."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402


def load(name: str) -> dict:
    p = pathlib.Path(name)
    if not p.is_dir():
        p = common.RESULTS / name
    return json.loads((p / "teacher_forced.json").read_text())


def diffs(a: list, b: list) -> dict:
    vals = [abs(float(x) - float(y)) for x, y in zip(a, b)
            if x is not None and y is not None and math.isfinite(float(x)) and math.isfinite(float(y))]
    return {"count": len(vals), "max_abs_diff": max(vals) if vals else None,
            "mean_abs_diff": statistics.mean(vals) if vals else None}


def load_forced(name: str) -> list[dict]:
    path = pathlib.Path(name)
    if not path.is_dir():
        path = common.RESULTS / name
    return [
        json.loads(line)
        for line in (path / "forced.jsonl").read_text().splitlines()
    ]


def quantile(values: list[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lo, hi = int(position), min(int(position) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def workload_distribution(a_name: str, b_name: str) -> dict:
    a = {row["key"]: row for row in load_forced(a_name)}
    b = {row["key"]: row for row in load_forced(b_name)}
    rows, problems = [], []
    for key in sorted(set(a) & set(b)):
        left, right = a[key], b[key]
        if left["forced_ids"] != right["forced_ids"]:
            problems.append(f"different forced ids for {key}")
        drift = diffs(left["logprobs"], right["logprobs"])
        rows.append({"key": key, **drift})
    missing = sorted(set(a) ^ set(b))
    if missing:
        problems.append(f"{len(missing)} requests present in only one receipt")
    means = [row["mean_abs_diff"] for row in rows if row["mean_abs_diff"] is not None]
    maxima = [row["max_abs_diff"] for row in rows if row["max_abs_diff"] is not None]
    def summary(values):
        return {
            "p10": quantile(values, 0.10),
            "p50": quantile(values, 0.50),
            "p90": quantile(values, 0.90),
            "mean": statistics.mean(values) if values else None,
            "max": max(values) if values else None,
        }
    return {
        "a": a_name,
        "b": b_name,
        "requests": len(rows),
        "per_request_mean_abs_diff": summary(means),
        "per_request_max_abs_diff": summary(maxima),
        "rows": rows,
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", type=pathlib.Path)
    ap.add_argument("--distribution", action="store_true",
                    help="compare forced.jsonl receipts and report per-request drift quantiles")
    args = ap.parse_args()
    if args.distribution:
        result = workload_distribution(args.a, args.b)
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        raise SystemExit(1 if result["problems"] else 0)
    a, b = load(args.a), load(args.b)
    problems = []
    for key in ("tp", "prompt_len", "k", "prompt_ids"):
        if a.get(key) != b.get(key):
            problems.append(f"different {key}")
    if a.get("generated_ids") != b.get("generated_ids"):
        problems.append("different generated continuation; teacher-forced labels are not the same")
    result = {
        "a": a["label"], "b": b["label"], "tp": a["tp"],
        "prompt": diffs(a["prompt_input_logprobs"], b["prompt_input_logprobs"]),
        "first_k_generated": diffs(a["forced_generated_token_logprobs"], b["forced_generated_token_logprobs"]),
        "generated_ids_equal": a.get("generated_ids") == b.get("generated_ids"),
        "problems": problems,
    }
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
