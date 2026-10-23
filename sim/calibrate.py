"""Calibrate the cost model on H200 measurements: every okc run's per-shape records in runs/.

For each problem, the latest run's SGLang times (baseline_ns, the judge's median after it subtracts
the graph replay's fixed cost) against the shape's speed of light (the judge's own bound, which on
memory-bound shapes is bytes / 4.8 TB/s), and a least-squares fit of SGLang's time per shape as
fixed + bytes / bandwidth on the memory-bound shapes.

    python -m sim.calibrate
"""

import json
from pathlib import Path

from sim import hw

REPO = Path(__file__).resolve().parent.parent


def latest_runs() -> dict[str, dict]:
    """problem -> the newest run JSON that has benchmark records."""
    out = {}
    for f in sorted((REPO / "runs").glob("*/*.json")):
        res = json.loads(f.read_text()).get("result") or {}
        if any(r.get("event") == "benchmark" for r in res.get("records", [])):
            out[f.parent.name] = {"file": f, **res}
    return out


def shapes(run: dict) -> list[dict]:
    rows = []
    for r in run["records"]:
        if r.get("event") != "benchmark" or r.get("baseline_ns") is None:
            continue
        rows.append({"case": r["case"], "sglang_us": r["baseline_ns"] / 1e3, "ours_us": r["median_ns"] / 1e3,
                     "sol_us": r["speed_of_light_ns"] / 1e3})
    return rows


def fit(rows: list[dict]) -> tuple[float, float]:
    """(fixed us, effective TB/s): least squares of sglang_us on sol_us, treating sol as bytes / 4.8 TB/s."""
    xs = [r["sol_us"] for r in rows]
    ys = [r["sglang_us"] for r in rows]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")
    return my - slope * mx, hw.HBM_BPS / 1e12 / slope


def main():
    for problem, run in sorted(latest_runs().items()):
        rows = shapes(run)
        print(f"== {problem}  ({run['file'].relative_to(REPO)}; {len(rows)} shapes)")
        print(f"   {'shape':<58}{'sglang':>9}{'sol':>9}{'sol/sglang':>11}")
        for r in rows:
            c = ", ".join(f"{k} {v}" for k, v in r["case"].items() if k not in ("seed", "weight"))
            print(f"   {c[:57]:<58}{r['sglang_us']:>9.2f}{r['sol_us']:>9.2f}{r['sol_us'] / r['sglang_us']:>11.0%}")
        big = [r for r in rows if r["sol_us"] >= 3.0]
        if len(big) >= 3:
            fixed, tbs = fit(big)
            print(f"   fit on {len(big)} shapes with sol >= 3 us: SGLang ~ {fixed:.2f} us + bytes / {tbs:.2f} TB/s")


if __name__ == "__main__":
    main()
