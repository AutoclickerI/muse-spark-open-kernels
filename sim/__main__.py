"""Print each problem's bounds and speedup ceilings against SGLang and the current leader.

    python -m sim               # live leaderboard (public API)
    python -m sim --offline     # no network: leaders left out
"""

import argparse
import json
import urllib.request

from sim import hw
from sim.problems import geomean, load_times, shape_bounds, spec, task_floors

PROBLEMS = ("muse_bf16_proj", "muse_attention_decode", "muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer")
API = "https://openkernelchallenge.dev/api/v1"


def kernels(problem: str, design: str) -> int:
    """Kernels per call in a design: 'multi' = a realistic split, 'mega' = one fused kernel."""
    if design == "mega":
        return 1
    return {"muse_bf16_proj": 1, "muse_attention_decode": 2, "muse_moe_w4a16": 2,
            "muse_decode_layer": 8, "muse_prefill_layer": 10}[problem]


def leader(problem: str) -> float | None:
    # The API refuses urllib's default User-Agent (403).
    req = urllib.request.Request(f"{API}/leaderboard/{problem}", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        rows = json.load(r)["rows"]
    return max(row["speedup"] for row in rows) if rows else None


def check_floors(problem: str) -> str:
    theirs = task_floors(problem)
    ours = [shape_bounds(problem, b, 1, tf)["floor"] for b, tf in zip(spec(problem)["benchmarks"], theirs)]
    gaps = sorted(((abs(a - b), a, b) for a, b in zip(ours, theirs)), reverse=True)
    return f"floors vs TASK.md ({len(theirs)} shapes, TASK rounds to 0.1 us): worst {gaps[0][1]:,.1f} vs {gaps[0][2]:,.1f} us"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    print(f"H200 model: {hw.HBM_BPS / 1e12:.1f} TB/s, {hw.BF16_FLOPS / 1e12:.0f} TFLOP/s BF16; practical: "
          f"{hw.EFF['bw']:.0%} of bandwidth, {hw.EFF['ramp_us']:.1f} us first kernel + {hw.EFF['gap_us']:.1f} us each further one")
    for p in PROBLEMS:
        s = spec(p)
        sglang = float(s["sglang_us"])
        multi, mega = load_times(p, kernels(p, "multi")), load_times(p, kernels(p, "mega"))
        print(f"\n== {p}  (SGLang {sglang:,.0f} us {s['sglang_per']}; {check_floors(p)})")
        print(f"   {'load':<12}{'floor':>12}{'expected':>12}{'practical':>12}{'megakernel':>12}   us per {s['sglang_per']}")
        for load in multi:
            m, g = multi[load], mega[load]
            print(f"   {load:<12}{m['floor']:>12,.0f}{m['expected']:>12,.0f}{m['practical']:>12,.0f}{g['practical']:>12,.0f}")
        gm = {k: geomean(v[k] for v in multi.values()) for k in ("floor", "expected", "practical")}
        gm["megakernel"] = geomean(v["practical"] for v in mega.values())
        line = "   ceiling vs SGLang: " + ", ".join(f"{k} {sglang / v:.2f}x" for k, v in gm.items())
        if not args.offline:
            best = leader(p)
            line += f"; leader now {best:.2f}x" if best else "; no leader"
        print(line)


if __name__ == "__main__":
    main()
