#!/usr/bin/env python3
"""Screen candidates in parallel and send each pass as soon as its screen finishes.

    tools/screen_all.py                       # screen problems/*/cand_*.py not yet sent; report
    tools/screen_all.py --send                # ... and send each pass through tools/okc_run.py
    tools/screen_all.py --jobs 8 --gpu 0 --skip-recent 20 muse_moe_w4a16 muse_decode_layer

A candidate is unsent when no receipt under runs/ (or runs/pending/) holds its exact source. Each
screen is `tools/local_eval.py` in the judge's image: every test, then CUDA-graph capture and replay
of every scored shape (2 replays, 1 input set, no baseline; under parallel load the times mean
nothing, the capture does). Each screen builds its extensions in a directory of its own, so
concurrent load_inline builds of same-named modules don't collide. A screen that ran out of GPU
memory beside the others is screened again alone. Files changed in the last --skip-recent minutes
are left to the lane still writing them.

An earlier screen of the same source (sha256) that passed everything is reused instead of rerun.

--send queues each pass with `tools/okc_run.py send` (receipts, priority queue, judge cap; a scored
run while the problem has one left today, else a check, else at the reset). Sends carry --lane and --decision, so PLAN ranks the decision independently of its host.
Results: runs/screen/<utc>/summary.json and one log per candidate.
"""

import argparse
import ast
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from okc_run import PRIORITY  # noqa: E402


def sent_hashes() -> set:
    seen = {hashlib.sha256(p.read_bytes()).hexdigest() for p in REPO.glob("runs/*/*.py")}
    for p in REPO.glob("runs/pending/*.json"):
        seen.add(json.loads(p.read_text()).get("sha256"))
    seen |= {w["sha256"] for w in waiting()}  # sends still in okc_run's local queue
    return seen


def waiting() -> list:
    """Live entries of okc_run's local queue (runs/waiting/<pid>.json)."""
    out = []
    for w in REPO.glob("runs/waiting/*.json"):
        try:
            e = json.loads(w.read_text())
            os.kill(e["pid"], 0)
        except (OSError, ValueError):
            continue
        out.append(e)
    return out


def unsent(problems, skip_recent_min):
    seen, now, out, recent = sent_hashes(), time.time(), [], []
    for prob in problems:
        for f in sorted((REPO / "problems" / prob).glob("cand_*.py")):
            if hashlib.sha256(f.read_bytes()).hexdigest() in seen:
                continue
            (recent if now - f.stat().st_mtime < skip_recent_min * 60 else out).append(f)
    return out, recent


def earlier_pass(f: Path):
    """An earlier screen's result for this exact source in which every test and graph passed."""
    sha, stem = hashlib.sha256(f.read_bytes()).hexdigest(), f"{f.parent.name}.{f.stem}"
    for js in sorted(REPO.glob(f"runs/screen/*/{stem}.json"), reverse=True):
        res = json.loads(js.read_text())
        checks = res.get("tests", []) + res.get("shapes", [])
        # a screen from before the static checks (undefined names, sm_90a compile) doesn't count
        if (res.get("sha256") == sha and checks and "static" in res
                and all(c.get("ok") for c in checks + res["static"])):
            return js, res
    return None


# The judge gets the one file plus the problem's official files (okc init), so an import of any other module
# of ours passes here and fails there.
OFFICIAL = {"reference", "baseline", "checks", "nvfp4"}


def local_imports(f: Path) -> set:
    ours = ({p.stem for p in REPO.glob("problems/**/*.py")} | {p.stem for p in REPO.glob("tools/*.py")}
            | {p.stem for p in REPO.glob("sim/*.py")} | {"sim", "prof", "tools"}) - OFFICIAL
    found = set()
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.Import):
            found |= {a.name for a in n.names if a.name.split(".")[0] in ours}
        elif isinstance(n, ast.ImportFrom) and (n.level or (n.module or "").split(".")[0] in ours):
            found.add("." * n.level + (n.module or ""))
    return found


def screen(f: Path, gpu: str, logdir: Path) -> dict:
    stem = f"{f.parent.name}.{f.stem}"
    log, js = logdir / f"{stem}.log", logdir / f"{stem}.json"
    if bad := local_imports(f):
        return {"problem": f.parent.name, "file": str(f.relative_to(REPO)), "ok": False, "rc": "local-import",
                "tests": "0/0", "graphs": "0/0", "prepare_s": None, "minutes": 0,
                "why": f"imports modules the judge won't have: {sorted(bad)}", "log": str(log.relative_to(REPO))}
    if done := earlier_pass(f):
        res = done[1]
        return {"problem": f.parent.name, "file": str(f.relative_to(REPO)), "ok": True, "rc": 0,
                "tests": f"{len(res['tests'])}/{len(res['tests'])}", "graphs": f"{len(res['shapes'])}/{len(res['shapes'])}",
                "prepare_s": res.get("prepare_s"), "minutes": 0, "why": "",
                "log": str(done[0].with_suffix(".log").relative_to(REPO))}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               # a path inside the container: tools/img maps the container's /tmp to its work dir on disk
               APPTAINERENV_TORCH_EXTENSIONS_DIR=f"/tmp/screen-ext/{stem}")
    t0 = time.time()
    with log.open("w") as out:
        try:
            rc = subprocess.run([str(REPO / "tools/img"), "python", "tools/local_eval.py", str(f.relative_to(REPO)),
                                 "--no-baseline", "--reps", "2", "--inputs", "1", "--json", str(js)],
                                cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=45 * 60).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    res = json.loads(js.read_text()) if js.exists() else {}
    tests = res.get("tests", [])
    shapes = res.get("shapes", [])
    why = ""
    if rc != 0:
        bad = ([t for t in tests if not t.get("ok")] + [s for s in shapes if not s.get("ok")]
               + [c for c in res.get("static", []) if not c.get("ok")])
        why = (bad[0].get("msg") or bad[0].get("error") or "") if bad else log.read_text()[-400:].strip().splitlines()[-1:]
        why = why if isinstance(why, str) else " ".join(why)
    return {"problem": f.parent.name, "file": str(f.relative_to(REPO)), "ok": rc == 0, "rc": rc,
            "tests": f"{sum(t.get('ok', False) for t in tests)}/{len(tests)}",
            "graphs": f"{sum(s.get('ok', False) for s in shapes)}/{len(shapes)}",
            "prepare_s": res.get("prepare_s"), "minutes": round((time.time() - t0) / 60, 1),
            "why": str(why)[:300], "log": str(log.relative_to(REPO))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", default=sorted(PRIORITY))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-recent", type=float, default=20, help="minutes")
    ap.add_argument("--send", action="store_true")
    ap.add_argument("--lane", default="P1", help="PLAN lane served by these screens (default P1)")
    ap.add_argument("--decision", default="screen unsent candidates", help="one-line decision for the ledger")
    args = ap.parse_args()

    files, recent = unsent(args.problems, args.skip_recent)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = REPO / "runs" / "screen" / stamp
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} unsent candidates to screen on GPU {args.gpu}, {args.jobs} at a time; "
          f"{len(recent)} skipped as still being written: {' '.join(p.name for p in recent)}", flush=True)
    def report(r):
        print(f"{'PASS' if r['ok'] else 'FAIL'} {r['file']} tests {r['tests']} graphs {r['graphs']}"
              f" ({r['minutes']} min){'  ' + r['why'] if not r['ok'] else ''}", flush=True)

    def send_pass(r):
        if not args.send or not r['ok']:
            return
        log = logdir / (Path(r['file']).stem + f".{r['problem']}.send.log")
        with log.open('w') as out:
            subprocess.Popen([sys.executable, str(REPO / "tools/okc_run.py"), "send", r['file'],
                              '--lane', args.lane, '--decision', args.decision,
                              '--note', f"screen_all {stamp}: local tests {r['tests']} and graphs {r['graphs']} pass"],
                             cwd=REPO, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        r['sent'] = True
        print(f"queued: {r['file']}", flush=True)

    results = []
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(screen, f, args.gpu, logdir) for f in files]
        for future in cf.as_completed(futures):
            r = future.result()
            report(r)
            results.append(r)
            send_pass(r)
    # Retry memory-contention failures alone. Successful peers have already entered the queue.
    for i, r in enumerate(results):
        log = REPO / r['log']
        if not r['ok'] and log.exists() and 'out of memory' in log.read_text():
            print(f"retry alone (out of memory in parallel): {r['file']}", flush=True)
            results[i] = screen(REPO / r['file'], args.gpu, logdir)
            report(results[i])
            send_pass(results[i])

    (logdir / "summary.json").write_text(json.dumps({"skipped_recent": [str(p.relative_to(REPO)) for p in recent],
                                                     "results": results}, indent=1))
    print(f"summary: {logdir.relative_to(REPO)}/summary.json  "
          f"pass {sum(r['ok'] for r in results)} / {len(results)}", flush=True)


if __name__ == "__main__":
    main()
