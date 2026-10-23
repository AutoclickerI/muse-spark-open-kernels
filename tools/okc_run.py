#!/usr/bin/env python3
"""Send a file to the judge with `okc check` or `okc submit`, and keep a receipt.

    tools/okc_run.py check  problems/muse_moe_w4a16/submission.py --note "split-K w2"
    tools/okc_run.py submit problems/muse_moe_w4a16/submission.py --note "..."
    tools/okc_run.py collect [job ...]     # finish receipts whose waiting process died
    tools/okc_run.py adopt <job> <file> --mode check|submit --note "..."   # a run sent without this tool

Each run leaves, under runs/<problem>/, <utc>_<mode>_<job>.py (the exact source sent) and
<utc>_<mode>_<job>.json (okc's whole result), and one line in runs/ledger.jsonl. okc itself
also writes profiles/<file>.<job>.txt and .log beside the file.

The judge's queue can take an hour or more. The source receipt and runs/pending/<job>.json are
written as soon as the judge accepts the job, before waiting, so a killed process loses nothing:
`collect` (no arguments: every pending job) waits on each with `okc status --wait` and writes the
rest. Running `collect` in the background is also the way to wait for results.

The judge starts our team's runs one at a time (FINDINGS, "Judge queue"), so at most OKC_INFLIGHT
(default 4) of our runs sit at the judge; further sends wait here in runs/waiting/ and go in
priority order (scored runs, then decode layer, MoE, attention and prefill, proj; --priority overrides),
woken by the judge finishing our oldest run or by another waiter leaving. OKC_INFLIGHT=0 sends at once.

Runs are capped per problem per day (10 scored, 20 checks; reset 00:00 UTC), and the judge
answers the same source again with its earlier result, so this refuses source the ledger
(or runs/pending/) already has for the same mode unless --force. Host Python, standard library only.
"""

import argparse
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "runs" / "ledger.jsonl"
PENDING = REPO / "runs" / "pending"
WAITING = REPO / "runs" / "waiting"
LOCK = REPO / "runs" / ".dispatch.lock"
CAP = int(os.environ.get("OKC_INFLIGHT", "4"))  # our runs at the judge at once; 0 sends at once
# Lower goes first; scored runs (-1) before every check. The order is docs/PLAN.md "Ranked work"
# (overall-board weight, FINDINGS "Boards").
PRIORITY = {"muse_decode_layer": 0, "muse_moe_w4a16": 1, "muse_bf16_proj": 2,
            "muse_attention_decode": 3, "muse_prefill_layer": 4}
JOB_KEYS = ("job_id", "job", "id")
OKC_MODE = {"check": "check", "submit": "run"}  # our mode -> okc's job mode


def okc_bin() -> str:
    return shutil.which("okc") or str(Path.home() / ".okc" / "bin" / "okc")


def okc_json(*args: str):
    proc = subprocess.run([okc_bin(), *args, "--json"], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout), proc
    except json.JSONDecodeError:
        return None, proc


def pick(d: dict, *keys):
    """The first of keys found at the top level or one level down (okc's JSON shape isn't documented)."""
    for k in keys:
        if k in d:
            return d[k]
    for v in d.values():
        if isinstance(v, dict):
            for k in keys:
                if k in v:
                    return v[k]
    return None


def git_rev() -> str:
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return (rev or "none") + ("+dirty" if dirty else "")


def newest_job(problem: str, mode: str, since: float):
    """The job okc accepted for us if --no-wait's reply didn't name it: our newest of that problem and mode."""
    jobs, _ = okc_json("jobs")
    jobs = [j for j in (jobs or []) if isinstance(j, dict) and j.get("problem") == problem
            and j.get("mode") == OKC_MODE[mode] and float(j.get("submitted_at") or 0) >= since - 5]
    return max(jobs, key=lambda j: float(j["submitted_at"]))["job_id"] if jobs else None


def known(sha: str, mode: str):
    """A ledger row or pending receipt for this source and mode, if there is one."""
    if LEDGER.exists():
        for line in LEDGER.read_text().splitlines():
            row = json.loads(line)
            if row["sha256"] == sha and row["mode"] == mode:
                return row
    for p in PENDING.glob("*.json") if PENDING.exists() else []:
        row = json.loads(p.read_text())
        if row["sha256"] == sha and row["mode"] == mode:
            return row
    for _, _, p in waiters():
        w = json.loads(p.read_text())
        if w["sha256"] == sha and w["mode"] == mode:
            return {"job": f"(waiting in {p.name})", "status": "waiting for a judge slot"}
    return None


def in_flight() -> list:
    """Our runs the judge hasn't finished (queued or running), oldest first: `okc jobs` is the authority,
    so runs sent by other tools or by processes that died still count."""
    jobs, _ = okc_json("jobs")
    live = [j for j in (jobs or []) if isinstance(j, dict) and j.get("status") in ("queued", "running")]
    return sorted(live, key=lambda j: float(j.get("submitted_at") or 0))


def waiters() -> list:
    """Sends waiting for a slot as (priority, time, path), best first; dead processes' entries are dropped."""
    out = []
    for p in WAITING.glob("*.json") if WAITING.exists() else []:
        try:
            w = json.loads(p.read_text())
            os.kill(w["pid"], 0)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        except ProcessLookupError:
            p.unlink(missing_ok=True)
            continue
        out.append((w["priority"], w["t"], p))
    return sorted(out)


def block_until_change(oldest_job):
    """Return when the oldest run in flight finishes (the judge's own completion signal) or a waiter
    leaves runs/waiting/ (inotify), whichever comes first."""
    done = threading.Event()

    def judge():
        okc_json("status", oldest_job, "--wait")
        done.set()

    def queue():
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_CLOEXEC)
        if fd >= 0 and libc.inotify_add_watch(fd, os.fsencode(WAITING), 0x200 | 0x40) >= 0:  # IN_DELETE, IN_MOVED_FROM
            os.read(fd, 4096)
            done.set()

    if oldest_job:
        threading.Thread(target=judge, daemon=True).start()
    threading.Thread(target=queue, daemon=True).start()
    done.wait()


def send_in_turn(entry: dict, priority: float, send):
    """Call send() once fewer than CAP of our runs are at the judge and no better waiter is ahead.
    The judge starts one of our runs at a time (FINDINGS, "Judge queue"), so holding the rest here
    costs no throughput and lets the most important run go next."""
    if CAP <= 0:
        return send()
    WAITING.mkdir(parents=True, exist_ok=True)
    me = WAITING / f"{os.getpid()}.json"
    me.write_text(json.dumps({"pid": os.getpid(), "priority": priority, "t": time.time(), **entry}))
    announced = False
    try:
        while True:
            with LOCK.open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                live = in_flight()
                ahead = waiters()
                if len(live) < CAP and ahead and ahead[0][2] == me:
                    result = send()
                    me.unlink(missing_ok=True)
                    return result
            if not announced:
                print(f"waiting for a judge slot: {len(live)} of our runs in flight (cap {CAP}), "
                      f"{len(ahead) - 1} other sends waiting", flush=True)
                announced = True
            block_until_change(live[0]["job_id"] if len(live) >= CAP else None)
    finally:
        me.unlink(missing_ok=True)


def start_receipt(problem, mode, job, src: bytes, file: Path, note, stamp, **extra) -> dict:
    out = REPO / "runs" / problem
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{stamp}_{mode}_{job}"
    stem.with_suffix(".py").write_bytes(src)
    row = {
        "utc": stamp, "problem": problem, "mode": mode, "job": job, "sha256": hashlib.sha256(src).hexdigest(),
        "file": os.path.relpath(file.resolve(), REPO), "source": str(stem.with_suffix(".py").relative_to(REPO)),
        "git": git_rev(), "note": note, **extra,
    }
    PENDING.mkdir(parents=True, exist_ok=True)
    (PENDING / f"{job}.json").write_text(json.dumps(row, indent=1))
    return row


def finish(job: str) -> dict:
    """Wait for a pending job's result, then write its .json and ledger row and drop the pending file."""
    pending = PENDING / f"{job}.json"
    if not pending.exists():
        print(f"{job}: no pending receipt (already in the ledger, or never sent through this tool)", flush=True)
        return {}
    row = json.loads(pending.read_text())
    while True:
        result, proc = okc_json("status", job, "--wait")
        if isinstance(result, dict) and result.get("status") not in (None, "queued", "running"):
            break
        print(f"okc status {job} --wait returned without a result ({proc.returncode}); waiting again", flush=True)
        time.sleep(30)  # only after okc itself gave up (network error), so we don't spin
    try:  # claim it, so two processes waiting on the same job write one ledger row
        claimed = pending.rename(pending.with_suffix(".done"))
    except FileNotFoundError:
        return {}
    REPO.joinpath(row["source"]).with_suffix(".json").write_text(json.dumps(result, indent=1, sort_keys=True))
    note = row.pop("note")
    row.update(status=pick(result, "status"), speedup=pick(result, "speedup"),
               interval=pick(result, "speedup_ci"), note=note)
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    claimed.unlink()
    print(json.dumps(row, indent=1), flush=True)
    return row


def send(args):
    src = args.file.read_bytes()
    m = re.match(rb"#\s*okc:\s*problem=(\w+)", src)
    if not m:
        sys.exit(f"{args.file}: the first line must be '# okc: problem=<name>'")
    problem = m.group(1).decode()
    row = None if args.force else known(hashlib.sha256(src).hexdigest(), args.mode)
    if row:
        sys.exit(f"this source already ran as {args.mode} job {row['job']} ({row.get('status', 'pending')}, "
                 f"speedup {row.get('speedup')}); --force to send it again")

    def send():
        # okc reads the file when it sends, so a file edited while this run waited would go out
        # under a receipt holding the old source.
        if args.file.read_bytes() != src:
            sys.exit(f"{args.file} changed while this run waited for a judge slot; not sent. "
                     "Send it again if the new source is what you want.")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sent_at = time.time()
        result, proc = okc_json(args.mode, problem, str(args.file), "--no-wait")
        job = pick(result, *JOB_KEYS) if isinstance(result, dict) else None
        job = job or newest_job(problem, args.mode, sent_at)
        if not job:
            sys.exit(f"okc {args.mode} failed ({proc.returncode}):\n{proc.stdout}{proc.stderr}")
        start_receipt(problem, args.mode, job, src, args.file, args.note, stamp)
        return job

    priority = args.priority if args.priority is not None else (
        -1 if args.mode == "submit" else PRIORITY.get(problem, 3))
    job = send_in_turn({"problem": problem, "mode": args.mode, "file": str(args.file),
                        "sha256": hashlib.sha256(src).hexdigest()}, priority, send)
    print(f"okc {args.mode} {problem} {args.file}: job {job} accepted; waiting for the result "
          f"(if this process dies, `tools/okc_run.py collect {job}` finishes the receipt)", flush=True)
    finish(job)


def collect(args):
    """Finish pending receipts in parallel, so each is written when its own run finishes."""
    jobs = args.jobs or sorted(f.stem for f in PENDING.glob("*.json"))
    threads = [threading.Thread(target=finish, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def adopt(args):
    """Receipt a run that was sent without this tool, from the file as it is now. The judge doesn't
    return the source, so if the file changed after the run was sent the row says source_uncertain."""
    src = args.file.read_bytes()
    m = re.match(rb"#\s*okc:\s*problem=(\w+)", src)
    status, proc = okc_json("status", args.job)
    if not m or not isinstance(status, dict):
        sys.exit(f"{args.file}: no '# okc: problem=' line, or okc status {args.job} failed:\n{proc.stdout}{proc.stderr}")
    problem = m.group(1).decode()
    created = float(status["created_at"])
    stamp = datetime.datetime.fromtimestamp(created, datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    extra: dict = {"adopted": True}
    if args.file.stat().st_mtime > created:
        extra["source_uncertain"] = "file changed after the run was sent"
    start_receipt(problem, args.mode, args.job, src, args.file, args.note, stamp, **extra)
    finish(args.job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for mode in ("check", "submit"):
        p = sub.add_parser(mode)
        p.add_argument("file", type=Path)
        p.add_argument("--note", default="", help="what this candidate is, for the ledger")
        p.add_argument("--force", action="store_true", help="send source the ledger already has")
        p.add_argument("--priority", type=float, help="queue order, lower first (default: by problem; scored runs -1)")
        p.set_defaults(mode=mode, run=send)
    p = sub.add_parser("collect", help="finish pending receipts (all of them if no job is named)")
    p.add_argument("jobs", nargs="*")
    p.set_defaults(run=collect)
    p = sub.add_parser("adopt", help="receipt a run sent without this tool")
    p.add_argument("job")
    p.add_argument("file", type=Path)
    p.add_argument("--mode", choices=["check", "submit"], required=True)
    p.add_argument("--note", default="")
    p.set_defaults(run=adopt)
    args = ap.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
