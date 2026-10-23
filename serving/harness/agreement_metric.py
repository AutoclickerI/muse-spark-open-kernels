#!/usr/bin/env python3
"""Compare a patch against the measured stock agreement and logprob spread.

The gate is deliberately distributional: it accepts a patch when its median
teacher-forced drift stays below stock's p90 and its token-agreement p10 does
not fall below stock's p10.  Exact-response counts are retained but never used
as the decision on near-tie fixtures.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

from compare import agreement_metric, load, compare
from compare_teacher_forced import load_forced, diffs, quantile


def flatten(values):
    return [value for row in values for value in row]


def distribution(rows):
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
        "requests": len(rows),
        "per_request_mean_abs_diff": summary(means),
        "per_request_max_abs_diff": summary(maxima),
    }


def forced_rows(a_name: str, b_name: str):
    a = {row["key"]: row for row in load_forced(a_name)}
    b = {row["key"]: row for row in load_forced(b_name)}
    return [
        {"key": key, **diffs(a[key]["logprobs"], b[key]["logprobs"])}
        for key in sorted(set(a) & set(b))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", action="append", required=True,
                        help="at least three stock output receipts")
    parser.add_argument("--patch", required=True)
    parser.add_argument("--forced-stock", action="append", required=True)
    parser.add_argument("--forced-patch", required=True)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()
    if len(args.stock) < 3 or len(args.forced_stock) < 3:
        parser.error("provide three or more stock receipts for a spread")

    output_pairs = []
    for i, left in enumerate(args.stock):
        for right in args.stock[i + 1:]:
            output_pairs.append(compare(*load(left), *load(right)))
    patch_pairs = [compare(*load(stock), *load(args.patch)) for stock in args.stock]
    stock_rows = flatten([pair["rows"] for pair in output_pairs])
    patch_rows = flatten([pair["rows"] for pair in patch_pairs])
    stock_output = agreement_metric(stock_rows)
    patch_output = agreement_metric(patch_rows)

    stock_forced_rows = flatten(
        [forced_rows(args.forced_stock[i], args.forced_stock[j])
         for i in range(len(args.forced_stock))
         for j in range(i + 1, len(args.forced_stock))]
    )
    patch_forced_rows = flatten(
        [forced_rows(stock, args.forced_patch) for stock in args.forced_stock]
    )
    stock_forced = distribution(stock_forced_rows)
    patch_forced = distribution(patch_forced_rows)
    stock_token_p10 = stock_output["token_agreement_rate"]["p10"]
    patch_token_p10 = patch_output["token_agreement_rate"]["p10"]
    stock_lp_p90 = stock_forced["per_request_mean_abs_diff"]["p90"]
    patch_lp_p50 = patch_forced["per_request_mean_abs_diff"]["p50"]
    stock_max_p90 = stock_forced["per_request_max_abs_diff"]["p90"]
    patch_max_p50 = patch_forced["per_request_max_abs_diff"]["p50"]
    result = {
        "stock_output_spread": stock_output,
        "patch_vs_stock_output": patch_output,
        "stock_teacher_forced_spread": stock_forced,
        "patch_vs_stock_teacher_forced": patch_forced,
        "gate": {
            "token_agreement_p10_within_stock_p10": patch_token_p10 >= stock_token_p10,
            "teacher_forced_mean_p50_within_stock_p90": patch_lp_p50 <= stock_lp_p90,
            "teacher_forced_max_p50_within_stock_p90": patch_max_p50 <= stock_max_p90,
        },
    }
    result["gate"]["pass"] = all(result["gate"].values())
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["gate"]["pass"] else 1)


if __name__ == "__main__":
    main()
