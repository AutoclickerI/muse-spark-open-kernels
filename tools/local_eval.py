#!/usr/bin/env python3
"""Local screen for a kernel submission: the problem's tests, then CUDA-graph timing against
SGLang's kernel (the problem's baseline.py), the way TASK.md's "How a run is timed" says the
judge does it: every timed replay on fresh inputs after an L2 flush, the two sides taking
turns in swapped order, a shape's speedup the median of paired ratios, the problem's speedup
the geometric mean over loads.

This box is not the judge. Its GPUs are sm_120; the judge's is an H200 (sm_90). A failed test
here is a real failure, a pass is a partial check (the judge uses its own seeds, and on the
layer problems its own router seed), and the timings only rank candidates loosely. H200
numbers come from `okc check` (tools/okc_run.py).

    tools/img python tools/local_eval.py problems/muse_moe_w4a16/submission.py
    tools/img python tools/local_eval.py <file> --tests-only
    tools/img python tools/local_eval.py <file> --loads "decode 16,verify 32" --no-baseline
"""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import re
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
FLUSH_BYTES = 1 << 30  # read before every timed replay; larger than any L2 here or on an H200
PREPARE_LIMIT_S = 300
# Modules built for Hopper only. Off sm_90 they exit the process from C++ ("no kernel image is
# available"), which no try/except catches, so a baseline that imports one isn't run there.
HOPPER_ONLY = ("sgl_kernel.flash_attn",)


def problem_of(path: Path) -> str:
    m = re.match(r"#\s*okc:\s*problem=(\w+)", path.read_text().splitlines()[0])
    if not m:
        sys.exit(f"{path}: the first line must be '# okc: problem=<name>'")
    return m.group(1)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def value(text: str):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def parse_case(line: str) -> dict:
    """'bs 128, sharding ep8, dist skewed' -> {'bs': 128, 'sharding': 'ep8', 'dist': 'skewed'}"""
    return {k: value(v.strip()) for k, v in (part.strip().split(" ", 1) for part in line.split(","))}


def tests_of(task_md: str) -> list[dict]:
    m = re.search(r"must pass these \d+ tests[^\n]*\n\n```\n(.*?)```", task_md, re.S)
    if not m:
        sys.exit("can't find the test list in TASK.md")
    return [parse_case(line) for line in m.group(1).strip().splitlines()]


def describe(case: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in case.items() if k not in ("seed", "weight", "load", "repeats"))


def clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: clone(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(clone(v) for v in obj)
    return obj


def copy_into(static, fresh):
    """Copy a fresh input set into the buffers a graph was captured on."""
    for k, v in fresh.items():
        if isinstance(v, torch.Tensor):
            if static[k].shape != v.shape:
                raise RuntimeError(f"input {k} changed shape between seeds: {tuple(static[k].shape)} vs {tuple(v.shape)}")
            static[k].copy_(v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            for s, f in zip(static[k], v):
                s.copy_(f)


class Side:
    """One implementation (the submission or the baseline): prepare once, then call(data)."""

    def __init__(self, name: str, mod, weights):
        self.name = name
        takes_state = len(inspect.signature(mod.kernel_fn).parameters) > 1
        t0 = time.perf_counter()
        if weights is None or not takes_state:
            self.call = mod.kernel_fn
        else:
            state = mod.prepare(clone(weights)) if hasattr(mod, "prepare") else clone(weights)
            self.call = lambda data: mod.kernel_fn(data, state)
        torch.cuda.synchronize()
        self.prepare_s = time.perf_counter() - t0


class Problem:
    def __init__(self, ref, weights):
        self.ref = ref
        self.weights = weights
        self.check_takes_weights = len(inspect.signature(ref.check_implementation).parameters) == 3

    def make_input(self, case: dict, seed: int):
        case = {**case, "seed": seed}
        if hasattr(self.ref, "case_fields"):  # the layer problems: this input's busiest expert range
            case.update(self.ref.case_fields(case, self.weights))
        return self.ref.generate_input(**case)

    def check(self, data, output) -> tuple[bool, str]:
        try:
            raw = self.ref.collect(data, output) if hasattr(self.ref, "collect") else output
        except ValueError as e:
            return False, str(e)
        args = (data, raw, self.weights) if self.check_takes_weights else (data, raw)
        return self.ref.check_implementation(*args)


def run_tests(problem: Problem, side: Side, tests: list[dict], seed0: int) -> list[dict]:
    results = []
    for i, case in enumerate(tests):
        row: dict = {"case": describe(case)}
        try:
            data = problem.make_input(case, seed0 + i)
            t0 = time.perf_counter()
            out = side.call(data)
            torch.cuda.synchronize()
            row["first_call_s"] = time.perf_counter() - t0
            row["ok"], row["msg"] = problem.check(data, out)
        except Exception as e:  # report and go on to the next case
            row["ok"], row["msg"] = False, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        print(f"  {'PASS' if row['ok'] else 'FAIL'}  {row['case']}: {row['msg']}", flush=True)
        results.append(row)
        torch.cuda.empty_cache()
    return results


def capture(call, data):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call(data)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(data)
    torch.cuda.synchronize()
    return graph, out


class Timer:
    def __init__(self):
        self.flush_buf = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        # A graph replay's fixed cost, subtracted from both sides as the judge does (~4 us there).
        x = torch.zeros(1, device="cuda")
        graph, _ = capture(lambda d: d.add_(1), x)
        self.fixed_us = statistics.median(self.replay_us(graph) for _ in range(50))

    def replay_us(self, graph) -> float:
        self.flush_buf.sum()  # cold L2: read a buffer larger than it
        self.start.record()
        graph.replay()
        self.end.record()
        self.end.synchronize()
        return self.start.elapsed_time(self.end) * 1e3


def time_shape(problem: Problem, sides: list[Side], case: dict, reps: int, n_inputs: int, timer: Timer) -> dict:
    sets = [problem.make_input(case, case["seed"] + 7919 * j) for j in range(n_inputs)]
    static = clone(sets[0])
    graphs = [capture(s.call, static) for s in sides]
    times = [[] for _ in sides]
    checks = []
    for r in range(reps):
        order = range(len(sides)) if r % 2 == 0 else reversed(range(len(sides)))
        for k in order:
            copy_into(static, sets[r % n_inputs])
            times[k].append(max(timer.replay_us(graphs[k][0]) - timer.fixed_us, 1e-3))
            if k == 0 and r < n_inputs:  # check the submission's replays, once per input set
                checks.append(problem.check(static, graphs[k][1]))
    row = {"case": describe(case), "load": case.get("load"), "weight": case.get("weight", 1),
           "us": statistics.median(times[0]), "ok": all(ok for ok, _ in checks)}
    if not row["ok"]:
        row["msg"] = next(msg for ok, msg in checks if not ok)
    if len(sides) > 1:
        # Paired ratios, two repetitions at a time with the order swapped.
        pairs = [(times[1][i] + times[1][i + 1]) / (times[0][i] + times[0][i + 1]) for i in range(0, reps - 1, 2)]
        row["speedup"] = statistics.median(pairs)
        row["base_us"] = statistics.median(times[1])
        row["us"] = row["base_us"] / row["speedup"]
        q = statistics.quantiles(pairs, n=4) if len(pairs) > 1 else [pairs[0]] * 3
        row["iqr"] = (q[0], q[2])
    del graphs, sets, static
    torch.cuda.empty_cache()
    return row


def summarize(rows: list[dict]) -> dict:
    """Each load's time is the weighted sum of its shapes' times; the problem's speedup is the
    geometric mean over loads of SGLang's load time over ours."""
    loads = {}
    for row in rows:
        acc = loads.setdefault(row["load"], {"us": 0.0, "base_us": 0.0})
        acc["us"] += row["weight"] * row["us"]
        if "base_us" in row:
            acc["base_us"] += row["weight"] * row["base_us"]
    for acc in loads.values():
        if acc["base_us"]:
            acc["speedup"] = acc["base_us"] / acc["us"]
    out: dict = {"loads": loads}
    if loads and all("speedup" in acc for acc in loads.values()):
        out["speedup"] = math.exp(statistics.fmean(math.log(acc["speedup"]) for acc in loads.values()))
    return out


def hopper_compile(file: Path) -> list[dict]:
    """Compile every CUDA source string in the file for sm_90a, whichever branch would build it.

    Hopper-only extensions never build here, so their compile errors otherwise reach the judge
    (cand_r3_v9all 83a8a755: a missing <ATen/cuda/CUDAContext.h>). A module-level string with an
    #include and a kernel or torch's extension header counts; one the file fills in at run time
    (.format/.replace/%) is skipped. Flags: torch's own for load_inline, plus the literal
    extra_cuda_cflags and extra_include_paths of the load_inline call that names the string."""
    import concurrent.futures as cf
    import subprocess
    import sysconfig
    import tempfile
    from torch.utils import cpp_extension

    tree = ast.parse(file.read_text())
    templated = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ("format", "replace", "format_map") and isinstance(n.value, ast.Name):
            templated.add(n.value.id)
        elif isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) and isinstance(n.left, ast.Name):
            templated.add(n.left.id)
    flags_of = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "load_inline":
            kw = {k.arg: k.value for k in n.keywords}
            src = kw.get("cuda_sources")
            names = [e.id for e in (src.elts if isinstance(src, (ast.List, ast.Tuple)) else [src]) if isinstance(e, ast.Name)]
            extra = []
            for key, prefix in (("extra_cuda_cflags", ""), ("extra_include_paths", "-I")):
                try:
                    extra += [prefix + str(v) for v in ast.literal_eval(kw[key])]
                except (KeyError, ValueError):
                    pass
            for nm in names:
                flags_of[nm] = extra
    sources = []
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            name, text = n.targets[0].id, n.value.value
            if "#include" in text and ("__global__" in text or "torch/extension.h" in text) and name not in templated:
                sources.append((name, text))
    base = (["-std=c++20", "-O0", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
             "-DTORCH_EXTENSION_NAME=hopper_check", "-DTORCH_API_INCLUDE_EXTENSION_H",
             "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
             "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
             "-isystem", sysconfig.get_paths()["include"]]
            + [a for p in cpp_extension.include_paths(device_type="cuda") for a in ("-isystem", p)])
    tmp = Path(tempfile.mkdtemp(prefix="hopper_compile_"))

    def one(item):
        name, text = item
        cu = tmp / f"{name}.cu"
        cu.write_text(text)
        extra = [f for f in flags_of.get(name, []) if not f.startswith(("-gencode", "-arch", "--generate-code"))]
        cmd = ["nvcc", *base, *extra, "-c", str(cu), "-o", str(cu.with_suffix(".o"))]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return {"name": name, "ok": False, "msg": "nvcc timed out (900 s)"}
        errs = [l for l in (p.stdout + p.stderr).splitlines() if "error" in l]
        return {"name": name, "ok": p.returncode == 0, "msg": " | ".join(errs[:3])[:400]}

    with cf.ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, sources))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="the submission (first line '# okc: problem=<name>')")
    ap.add_argument("--tests-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true", help="skip the tests")
    ap.add_argument("--no-baseline", action="store_true", help="time the submission alone")
    ap.add_argument("--loads", help="comma-separated loads to time (default: all)")
    ap.add_argument("--reps", type=int, default=20, help="timed replays per side per shape (even)")
    ap.add_argument("--inputs", type=int, default=3, help="fresh input sets per shape")
    ap.add_argument("--seed", type=int, default=90000, help="first test seed (the judge's are its own)")
    ap.add_argument("--json", type=Path, help="write the result here")
    args = ap.parse_args()

    file = args.file.resolve()
    name = problem_of(file)
    pdir = REPO / "problems" / name
    sys.path[:0] = [str(pdir), str(file.parent)]  # reference.py imports checks and nvfp4 by name
    ref = load_module("reference", pdir / "reference.py")
    spec = json.loads((REPO / "upstream" / "api" / "problems" / f"{name}.json").read_text())
    tests, benches = tests_of((pdir / "TASK.md").read_text()), spec["benchmarks"]
    if args.loads:
        wanted = {s.strip() for s in args.loads.split(",")}
        benches = [b for b in benches if b["load"] in wanted]

    dev = torch.cuda.get_device_properties(0)
    print(f"{name}: {file.relative_to(REPO) if file.is_relative_to(REPO) else file}")
    print(f"LOCAL screen on {dev.name} (sm_{dev.major}{dev.minor}); the judge runs an H200 (sm_90),"
          " so these timings are not its timings")
    result = {"problem": name, "file": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
              "gpu": dev.name, "cc": f"{dev.major}.{dev.minor}", "torch": torch.__version__}

    if (dev.major, dev.minor) != (9, 0) and hasattr(ref, "_fa3_scheduler_metadata"):
        # generate_input asks FA3 for its tile schedule on full-attention decode steps; FA3 imports
        # here but can't run. Take the reference's own no-FA3 path (it returns None on ImportError).
        setattr(ref, "_fa3_scheduler_metadata", lambda *_, **__: None)
        print("note: off Hopper, inputs carry scheduler_metadata=None (the judge's carry FA3's schedule)")
    weights = ref.generate_weights(ref.WEIGHTS_SEED) if hasattr(ref, "generate_weights") else None
    if hasattr(ref, "weights_fields"):
        print("note: the judge also picks a router seed per run (weights_fields) to hit each load's"
              " target MoE load; this uses the default router")
    problem = Problem(ref, weights)
    sub = Side("submission", load_module("submission", file), weights)
    result["prepare_s"] = sub.prepare_s
    print(f"prepare: {sub.prepare_s:.1f} s" + (f"  (over the {PREPARE_LIMIT_S} s limit)" if sub.prepare_s > PREPARE_LIMIT_S else ""))

    if not args.bench_only:
        print(f"tests ({len(tests)}):")
        result["tests"] = run_tests(problem, sub, tests, args.seed)
        if not all(t["ok"] for t in result["tests"]):
            print("a test failed; not timing")
            args.tests_only = True
    if not args.tests_only:
        sides = [sub]
        hopper_only = [m for m in HOPPER_ONLY if m in (pdir / "baseline.py").read_text()]
        if hopper_only and (dev.major, dev.minor) != (9, 0) and not args.no_baseline:
            print(f"baseline uses {', '.join(hopper_only)}, built for Hopper only; timing the submission alone")
        elif not args.no_baseline:
            try:
                sides.append(Side("baseline", load_module("baseline", pdir / "baseline.py"), weights))
            except Exception as e:
                print(f"baseline unavailable here ({type(e).__name__}: {e}); timing the submission alone")
        timer = Timer()
        print(f"timing {len(benches)} shapes, {args.reps} replays a side, {args.inputs} input sets"
              f" (graph replay fixed cost {timer.fixed_us:.1f} us, subtracted):")
        rows = []
        for case in benches:
            try:
                row = time_shape(problem, sides, case, args.reps, args.inputs, timer)
            except Exception as e:
                traceback.print_exc()
                row = {"case": describe(case), "load": case.get("load"), "ok": False, "msg": f"{type(e).__name__}: {e}"}
                print(f"  FAIL  [{row['load']}] {row['case']}: {row['msg']}")
                rows.append(row)
                continue
            line = f"  {'ok  ' if row['ok'] else 'FAIL'}  [{row['load']}] {row['case']}: {row['us']:.1f} us"
            if "speedup" in row:
                line += f" vs SGLang {row['base_us']:.1f} us = {row['speedup']:.3f}x (IQR {row['iqr'][0]:.3f}-{row['iqr'][1]:.3f})"
            print(line + ("" if row["ok"] else f"  -- {row['msg']}"), flush=True)
            rows.append(row)
        result["shapes"] = rows
        if all("us" in r for r in rows):
            result.update(summarize(rows))
            for load, acc in result["loads"].items():
                print(f"  load {load}: {acc['us']:.1f} us" + (f", {acc['speedup']:.3f}x" if "speedup" in acc else ""))
            if "speedup" in result:
                print(f"LOCAL speedup (geomean over loads): {result['speedup']:.3f}x  [not an H200 number]")

    if (dev.major, dev.minor) != (9, 0):
        result["hopper_compile"] = hopper_compile(file)
        bad = [c for c in result["hopper_compile"] if not c["ok"]]
        print(f"sm_90a compile of {len(result['hopper_compile'])} CUDA sources: "
              + ("all pass" if not bad else "; ".join(f"FAIL {c['name']}: {c['msg']}" for c in bad)), flush=True)

    if args.json:
        args.json.write_text(json.dumps(result, indent=1, default=str))
    ok = (all(t["ok"] for t in result.get("tests", [])) and all(r.get("ok") for r in result.get("shapes", []))
          and all(c["ok"] for c in result.get("hopper_compile", [])))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
