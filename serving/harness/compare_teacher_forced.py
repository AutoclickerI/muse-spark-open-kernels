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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()
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
