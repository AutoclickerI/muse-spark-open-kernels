"""Expected MoE work on the busiest GPU: distinct local experts touched and (token, expert) pairs,
sampled from muse_moe_w4a16's own generate_input (its routing distribution and its rotation that
makes GPU 0 the busiest). Writes sim/routing.json, which the model reads; torch is needed only here.

    python -m sim.routing [--samples 200]
"""

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).with_name("routing.json")
TOKENS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def load_reference():
    pdir = REPO / "problems" / "muse_moe_w4a16"
    sys.path.insert(0, str(pdir))
    spec = importlib.util.spec_from_file_location("moe_reference", pdir / "reference.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    args = ap.parse_args()
    import torch

    ref = load_reference()
    table = {}
    for t in TOKENS:
        distinct, pairs = [], []
        for s in range(args.samples):
            ids = ref.generate_input(bs=t, sharding="ep8", seed=10_000 + s, device="cpu")["topk_ids"]
            local = ids[ids < ref.LOCAL["ep8"][1]]
            distinct.append(int(torch.unique(local).numel()))
            pairs.append(int(local.numel()))
        table[t] = {"distinct": statistics.fmean(distinct), "pairs": statistics.fmean(pairs)}
        print(f"tokens {t:5d}: {table[t]['distinct']:6.2f} local experts touched, {table[t]['pairs']:7.1f} pairs")
    OUT.write_text(json.dumps({"source": "muse_moe_w4a16 reference.generate_input, dist normal, rank busiest",
                               "samples": args.samples, "tokens": table}, indent=1))


def expected(tokens: float) -> tuple[float, float]:
    """(distinct experts, pairs) on the busiest GPU, interpolated in log(tokens)."""
    import math
    table = {int(k): v for k, v in json.loads(OUT.read_text())["tokens"].items()}
    ks = sorted(table)
    if tokens <= ks[0]:
        return table[ks[0]]["distinct"], table[ks[0]]["pairs"] * tokens / ks[0]
    for lo, hi in zip(ks, ks[1:]):
        if tokens <= hi:
            w = (math.log(tokens) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return tuple(table[lo][k] * (1 - w) + table[hi][k] * w for k in ("distinct", "pairs"))
    return table[ks[-1]]["distinct"], table[ks[-1]]["pairs"] * tokens / ks[-1]


if __name__ == "__main__":
    main()
