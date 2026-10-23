#!/usr/bin/env python3
"""Screen every candidate the judge hasn't measured, in parallel, then send each one that passes.

    tools/screen_all.py                       # screen problems/*/cand_*.py not yet sent; report
    tools/screen_all.py --send                # ... and send each pass through tools/okc_run.py
    tools/screen_all.py --jobs 8 --gpu 0 --skip-recent 20 muse_moe_w4a16 muse_decode_layer

A candidate is unsent when no receipt under runs/ (or runs/pending/) holds its exact source. Each
screen is `tools/local_eval.py` in the judge's image: every test, then CUDA-graph capture and replay
of every scored shape (2 replays, 1 input set, no baseline; under parallel load the times mean
nothing, the capture does). Each screen builds its extensions in a directory of its own, so
concurrent load_inline builds of same-named modules don't collide. Files changed in the last
--skip-recent minutes are left to the lane still writing them.

--send queues each pass through tools/okc_run.py (receipts, priority queue, judge cap): a check while
the problem has checks left today (`okc problems`), else a scored run while it has runs left (the
board keeps a team's best run, so a slower scored run costs no place), else nothing (reported). Sends
take the problem's default priority + 0.5, so lanes' own later sends at the default go first.
Results: runs/screen/<utc>/summary.json and one log per candidate.
"""

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRIORITY = {"muse_decode_layer": 0, "muse_moe_w4a16": 1, "muse_attention_decode": 2,
            "muse_prefill_layer": 2, "muse_bf16_proj": 3}
WORK = Path(os.environ.get("MUSE_IMG_WORK", "/data2/jihoon/muse_kernel/img-work"))


def sent_hashes() -> set:
    seen = {hashlib.sha256(p.read_bytes()).hexdigest() for p in REPO.glob("runs/*/*.py")}
    for p in REPO.glob("runs/pending/*.json"):
        seen.add(json.loads(p.read_text()).get("sha256"))
    return seen


def unsent(problems, skip_recent_min):
    seen, now, out, recent = sent_hashes(), time.time(), [], []
    for prob in problems:
        for f in sorted((REPO / "problems" / prob).glob("cand_*.py")):
            if hashlib.sha256(f.read_bytes()).hexdigest() in seen:
                continue
            (recent if now - f.stat().st_mtime < skip_recent_min * 60 else out).append(f)
    return out, recent


def screen(f: Path, gpu: str, logdir: Path) -> dict:
    stem = f"{f.parent.name}.{f.stem}"
    log, js = logdir / f"{stem}.log", logdir / f"{stem}.json"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
               APPTAINERENV_TORCH_EXTENSIONS_DIR=str(WORK / "screen-ext" / stem))
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
        bad = [t for t in tests if not t.get("ok")] + [s for s in shapes if not s.get("ok")]
        why = (bad[0].get("msg") or bad[0].get("error") or "") if bad else log.read_text()[-400:].strip().splitlines()[-1:]
        why = why if isinstance(why, str) else " ".join(why)
    return {"problem": f.parent.name, "file": str(f.relative_to(REPO)), "ok": rc == 0, "rc": rc,
            "tests": f"{sum(t.get('ok', False) for t in tests)}/{len(tests)}",
            "graphs": f"{sum(s.get('ok', False) for s in shapes)}/{len(shapes)}",
            "prepare_s": res.get("prepare_s"), "minutes": round((time.time() - t0) / 60, 1),
            "why": str(why)[:300], "log": str(log.relative_to(REPO))}


def quotas() -> dict:
    out = subprocess.run([os.path.expanduser("~/.okc/bin/okc"), "problems"], capture_output=True, text=True).stdout
    q = {}
    for m in re.finditer(r"^(muse_\w+).*?(\d+) of \d+ runs left today, (\d+) of \d+ checks", out, re.M):
        q[m.group(1)] = {"submit": int(m.group(2)), "check": int(m.group(3))}
    # sends still waiting in okc_run's local queue have not reached okc's count yet
    for w in REPO.glob("runs/waiting/*.json"):
        try:
            argv = Path(f"/proc/{w.stem}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        mode = "check" if b"check" in argv else "submit" if b"submit" in argv else None
        prob = json.loads(w.read_text()).get("problem")
        if mode and prob in q:
            q[prob][mode] -= 1
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="*", default=sorted(PRIORITY))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--skip-recent", type=float, default=20, help="minutes")
    ap.add_argument("--send", action="store_true")
    args = ap.parse_args()

    files, recent = unsent(args.problems, args.skip_recent)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logdir = REPO / "runs" / "screen" / stamp
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} unsent candidates to screen on GPU {args.gpu}, {args.jobs} at a time; "
          f"{len(recent)} skipped as still being written: {' '.join(p.name for p in recent)}", flush=True)
    results = []
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        for r in pool.map(lambda f: screen(f, args.gpu, logdir), files):
            print(f"{'PASS' if r['ok'] else 'FAIL'} {r['file']} tests {r['tests']} graphs {r['graphs']}"
                  f" ({r['minutes']} min){'  ' + r['why'] if not r['ok'] else ''}", flush=True)
            results.append(r)

    if args.send:
        q = quotas()
        for r in sorted((r for r in results if r["ok"]), key=lambda r: PRIORITY.get(r["problem"], 3)):
            left = q.get(r["problem"], {})
            mode = "check" if left.get("check", 0) > 0 else "submit" if left.get("submit", 0) > 0 else None
            if not mode:
                r["sent"] = "no quota left today"
                continue
            left[mode] -= 1
            log = logdir / (Path(r["file"]).stem + f".{r['problem']}.send.log")
            subprocess.Popen([sys.executable, str(REPO / "tools/okc_run.py"), mode, r["file"],
                              "--priority", str(PRIORITY.get(r["problem"], 3) + 0.5),
                              "--note", f"screen_all {stamp}: untested candidate, local tests {r['tests']} and graphs {r['graphs']} pass"],
                             cwd=REPO, stdout=log.open("w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
            r["sent"] = mode
            print(f"sent {mode}: {r['file']}", flush=True)

    (logdir / "summary.json").write_text(json.dumps({"skipped_recent": [str(p.relative_to(REPO)) for p in recent],
                                                     "results": results}, indent=1))
    print(f"summary: {logdir.relative_to(REPO)}/summary.json  "
          f"pass {sum(r['ok'] for r in results)} / {len(results)}", flush=True)


if __name__ == "__main__":
    main()
