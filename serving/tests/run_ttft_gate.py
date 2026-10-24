"""Uninstrumented, bracketed TTFT/agreement screen at all four serving loads."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/data2/jihoon/muse_kernel/harness")


def run(label, tree, tp, native=False):
    command = [
        sys.executable,
        "serving/tests/integration_harness.py",
        "--with-logprob",
        "--tree",
        str(tree),
        "--weights",
        "margin",
        "--label",
        label,
        "--tp",
        str(tp),
        "--gpus",
        "0,1" if tp == 2 else "0",
        "--levels",
        "1,4,16,32",
        "--convs",
        "1:2,4:8,16:16,32:32",
        "--turns",
        "3",
    ]
    if native:
        command.append("--native")
    print("RUN", label, flush=True)
    with (DATA / "logs" / f"{label}.log").open("w") as out:
        subprocess.run(
            command, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True
        )
    meta = json.loads((DATA / "results" / label / "meta.json").read_text())
    assert not meta.get("failure") and not meta["validation_errors"], label
    assert meta["provenance"] == meta["provenance_end"], label


def compare(stock, patch, out):
    dest = out / f"{stock}--{patch}.json"
    with dest.with_suffix(".txt").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "serving/harness/compare.py",
                stock,
                patch,
                "--require-exact",
                "--json",
                str(dest),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    drift = out / f"{stock}--{patch}-drift.json"
    subprocess.run(
        [
            sys.executable,
            "serving/harness/compare_teacher_forced.py",
            stock,
            patch,
            "--json",
            str(drift),
        ],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    data = json.loads(dest.read_text())
    ratio = {
        c: data["per_level"][str(c)]["ttft_ratio_b_over_a"] for c in (1, 4, 16, 32)
    }
    print(stock, patch, ratio, flush=True)
    return {
        "stock": stock,
        "patch": patch,
        "ttft_p50_ratios": ratio,
        "target_1_5x": all(x <= 1.5 for x in ratio.values()),
        "hard_2x": all(x <= 2 for x in ratio.values()),
        "drift": json.loads(drift.read_text()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    out = DATA / "results" / f"{args.prefix}-evidence"
    out.mkdir(exist_ok=False)
    rows = []
    for tp in (2, 1):
        stocks = [f"{args.prefix}-stock-{v}-tp{tp}" for v in ("a", "b")]
        patches = [f"{args.prefix}-{v}-tp{tp}" for v in ("native", "portable")]
        run(stocks[0], "stock", tp)
        run(patches[0], args.tree, tp, native=True)
        run(patches[1], args.tree, tp)
        run(stocks[1], "stock", tp)
        for stock in stocks:
            for patch in patches:
                rows.append(compare(stock, patch, out))
        rows.append(compare(stocks[0], stocks[1], out))
        (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    patches = [r for r in rows if "-stock-" not in r["patch"]]
    assert all(r["target_1_5x"] for r in patches), (
        "TTFT margin target failed; retain every run"
    )
    print("ALL MODES/LOADS <=1.5x, EXACT220, STABLE PROVENANCE", flush=True)
