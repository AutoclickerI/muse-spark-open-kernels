"""Full-grid diagnostic preserving the warmup/cache history before c16."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/data2/jihoon/muse_kernel/harness")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    args = p.parse_args()
    for kind, tree in [
        ("stock", "stock"),
        ("before", args.before),
        ("after", args.after),
    ]:
        label = f"{args.prefix}-{kind}-tp2"
        print("TRACE", label, flush=True)
        cmd = [
            sys.executable,
            "serving/tests/ttft_harness.py",
            "--native",
            "--trace-ttft",
            "--tree",
            str(tree),
            "--weights",
            "margin",
            "--label",
            label,
            "--tp",
            "2",
            "--gpus",
            "0,1",
            "--levels",
            "1,4,16,32",
            "--convs",
            "1:2,4:8,16:16,32:32",
            "--turns",
            "3",
        ]
        with (DATA / "logs" / f"{label}.log").open("w") as out:
            subprocess.run(
                cmd, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, check=True
            )
