"""Summarize steady full-batch cadence using CUPTI kernel interval unions."""
import argparse
import bisect
import collections
import json
from pathlib import Path
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("sqlite")
ap.add_argument("--output")
ap.add_argument("--ab-mode", choices=["baseline", "patch"])
args = ap.parse_args()
rows = [json.loads(line) for line in Path(args.trace).read_text().splitlines()]
db = sqlite3.connect(args.sqlite)
offset = db.execute("select systemClockNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
capture_end = db.execute("select max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
kernels = db.execute("select start,end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
for table in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
    if table in tables:
        kernels += db.execute(f"select start,end from {table}").fetchall()
kernels.sort()
anchor = next((r for r in rows if r["kind"] == "anchor"), None)
if anchor:
    # Nsight graph-level tracing omits child kernels. Add asynchronously
    # measured graph envelopes, calibrated before requests begin. Their event
    # interval includes device scheduling gaps, so busy is an upper bound.
    base = (anchor["before_ns"] + anchor["after_ns"]) // 2 - offset
    kernels += [(base + int(r["device_start_ms"]*1e6), base + int(r["device_end_ms"]*1e6))
                for r in rows if r["kind"] == "graph_device"]
    kernels.sort()
merged = []
for a, b in kernels:
    if merged and a <= merged[-1][1]:
        merged[-1][1] = max(b, merged[-1][1])
    else:
        merged.append([a, b])
ends = [b for a, b in merged]
hosts = [r for r in rows if r["kind"] == "host"]
builds = [r for r in hosts if r["name"] == "get_next_batch_to_run"]
build_starts = [r["start_ns"] for r in builds]
runs = [r for r in hosts if r["name"] == "run_batch"]
result = dict(definition="Median steady decode; wall is build-to-next-build cadence; busy is the union of CUPTI kernel/copy/memset intervals", config=[r for r in rows if r["kind"] == "config"])
if anchor:
    result["definition"] = "Median steady decode cadence; busy is the union of CUPTI eager kernels and CUDA-event graph envelopes (upper bound); gap is the complement (lower bound)"
    result["clock_calibration_uncertainty_ms"] = (anchor["after_ns"] - anchor["before_ns"]) / 2e6
for bs in (1, 4):
    groups = collections.defaultdict(list)
    for i in range(2, len(runs)-1):
        run = runs[i]
        if args.ab_mode and run.get("ab_mode") != args.ab_mode:
            continue
        if args.ab_mode and not all(r.get("ab_mode") == args.ab_mode for r in runs[i-2:i+2]):
            continue
        if not all(r["mode"] == "2" and r["bs"] == bs for r in runs[i-2:i+2]):
            continue
        j = bisect.bisect_right(build_starts, run["start_ns"]) - 1
        if j < 0 or j+1 >= len(builds):
            continue
        a, b = builds[j]["start_ns"]-offset, builds[j+1]["start_ns"]-offset
        if b > capture_end:
            continue
        busy = 0
        k = bisect.bisect_left(ends, a)
        while k < len(merged) and merged[k][0] < b:
            x, y = merged[k]
            busy += max(0, min(b, y)-max(a, x))
            k += 1
        groups["scheduler_cadence_ms"].append((b-a)/1e6)
        groups["gpu_busy_ms"].append(busy/1e6)
        groups["gpu_gap_ms"].append((b-a-busy)/1e6)
        groups["run_batch_ms"].append(run["host_ms"])
        groups["batch_build_ms"].append(builds[j]["host_ms"])
        for row in hosts:
            if row["step"] == run["step"] and row["name"] not in ("run_batch", "get_next_batch_to_run", "update_running_batch"):
                groups[row["name"]+"_ms"].append(row["host_ms"])
    result[str(bs)] = dict(steps=len(groups["run_batch_ms"]), **{k: round(statistics.median(v), 6) for k, v in groups.items() if v})
print(json.dumps(result, indent=2))
if args.output:
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
