#!/usr/bin/env python3
"""Send a file to the judge, as a scored run while the problem has one left today, and keep a receipt.

    tools/okc_run.py send problems/muse_moe_w4a16/cand_v12.py --note "split-K w2"
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
priority order (PRIORITY; --priority overrides), woken by the judge finishing our oldest run or by
another waiter leaving. OKC_INFLIGHT=0 sends at once.

The mode is chosen when the run goes out: a scored run while the problem has one left today
(`okc problems`), else a check. A check and a scored run are the same run, and the board keeps a
team's best scored run, so a slower scored run costs no place and a faster one needs no second run.
With neither left, the send leaves the queue order until the daily reset (00:00 UTC) and then goes.
A source that already ran as a check goes again only as a scored run. The judge answers the same
source again with its earlier result, so this refuses a source that already has a scored run, or
is pending or waiting, unless --force. Host Python, standard library only.
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
import signal
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
# Lower goes first. The order is docs/PLAN.md "Ranked work" (overall-board weight, FINDINGS "Boards").
PRIORITY = {"muse_decode_layer": 0, "muse_moe_w4a16": 1, "muse_bf16_proj": 2,
            "muse_attention_decode": 3, "muse_prefill_layer": 4}
JOB_KEYS = ("job_id", "job", "id")
OKC_MODE = {"check": "check", "submit": "run"}  # our mode -> okc's job mode


def okc_bin() -> str:
    return shutil.which("okc") or str(Path.home() / ".okc" / "bin" / "okc")


def okc_json(*args: str):
    # dies with this process (PR_SET_PDEATHSIG), so a killed okc_run leaves no `okc status --wait` polling
    proc = subprocess.run([okc_bin(), *args, "--json"], capture_output=True, text=True,
                          preexec_fn=lambda: ctypes.CDLL(None).prctl(1, signal.SIGTERM))
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


def known(sha: str):
    """The modes this source already ran in (ledger), and a pending or waiting entry for it, if any."""
    ran = {row["mode"] for line in (LEDGER.read_text().splitlines() if LEDGER.exists() else [])
           if (row := json.loads(line))["sha256"] == sha}
    for p in PENDING.glob("*.json") if PENDING.exists() else []:
        row = json.loads(p.read_text())
        if row["sha256"] == sha:
            return ran, row
    for p, w in entries():
        if w["sha256"] == sha:
            return ran, {"job": f"(waiting in {p.name})", "status": "waiting for a judge slot"}
    return ran, None


def runs_left(problem: str) -> dict:
    """Today's scored runs and checks left for problem: the judge's own count (`okc problems`)."""
    d, proc = okc_json("problems")
    for p in (d or {}).get("problems", []):
        if p.get("name") == problem:
            return {"submit": p["limits"]["scored"]["left"], "check": p["limits"]["checks"]["left"]}
    sys.exit(f"okc problems has no {problem}:\n{proc.stdout}{proc.stderr}")


def next_reset() -> float:
    """The daily caps reset at 00:00 UTC."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def in_flight() -> list:
    """Our runs the judge hasn't finished (queued or running), oldest first: `okc jobs` is the authority,
    so runs sent by other tools or by processes that died still count."""
    jobs, proc = okc_json("jobs")
    while not isinstance(jobs, list):  # a failed call is not an empty judge: sending on it overfills the cap
        print(f"okc jobs failed ({proc.returncode}); asking again in 15 s", flush=True)
        time.sleep(15)
        jobs, proc = okc_json("jobs")
    live = [j for j in jobs if isinstance(j, dict) and j.get("status") in ("queued", "running")]
    return sorted(live, key=lambda j: float(j.get("submitted_at") or 0))


def entries() -> list:
    """Live sends in runs/waiting/ as (path, entry), including those held until the reset; dead
    processes' entries are dropped."""
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
        out.append((p, w))
    return out


def waiters() -> list:
    """Sends waiting for a slot as (priority, time, path), best first; those held until the reset are out."""
    return sorted((w["priority"], w["t"], p) for p, w in entries() if w.get("until", 0) <= time.time())


CHANGED = threading.Event()  # the oldest run in flight finished, or a waiter left the queue
WATCHING = set()  # what this process already watches: "queue" and the jobs with a judge thread


def block_until_change(oldest_job):
    """Return when the oldest run in flight finishes (the judge's own completion signal) or a waiter
    leaves the queue (it sends SIGUSR1 to the rest: see send_in_turn), whichever comes first.
    No inotify: dozens of waiters each holding an instance exhausted the per-user limit of 128."""
    if "queue" not in WATCHING:
        def queue():
            while True:
                signal.sigwait({signal.SIGUSR1})
                CHANGED.set()
        threading.Thread(target=queue, daemon=True).start()
        WATCHING.add("queue")
    if oldest_job and oldest_job not in WATCHING:
        def judge():
            okc_json("status", oldest_job, "--wait")
            WATCHING.discard(oldest_job)  # if it returned early, the next look watches it again
            CHANGED.set()
        threading.Thread(target=judge, daemon=True).start()
        WATCHING.add(oldest_job)
    CHANGED.wait()
    CHANGED.clear()


def wake_waiters():
    """Tell every other okc_run waiting for a slot that the queue changed."""
    for p, _ in entries():
        pid = int(p.stem)
        try:
            if pid != os.getpid() and b"okc_run.py" in Path(f"/proc/{pid}/cmdline").read_bytes():
                os.kill(pid, signal.SIGUSR1)
        except OSError:
            continue


def send_in_turn(entry: dict, priority: float, send, mode_now):
    """Call send(mode) once fewer than CAP of our runs are at the judge, no better waiter is ahead and
    mode_now() names a mode with runs left today; with none left, wait out of the queue order until
    the reset. The judge starts one of our runs at a time (FINDINGS, "Judge queue"), so holding the
    rest here costs no throughput and lets the most important run go next."""
    WAITING.mkdir(parents=True, exist_ok=True)
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGUSR1})  # before any thread starts: sigwait receives it
    me = WAITING / f"{os.getpid()}.json"
    mine = {"pid": os.getpid(), "priority": priority, "t": time.time(), **entry}
    me.write_text(json.dumps(mine))
    announced = False
    try:
        while True:
            # Only the head of the queue asks the judge (`okc jobs`) and watches its oldest run; the
            # rest read runs/waiting/ and sleep until the queue changes, so an event costs one API call.
            ahead = waiters()
            place = next(i for i, w in enumerate(ahead) if w[2] == me)
            live, until = [], 0
            if place == 0:
                with LOCK.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    live = in_flight()
                    if (CAP <= 0 or len(live) < CAP) and waiters()[0][2] == me:
                        mode = mode_now()
                        if mode:
                            result = send(mode)
                            me.unlink(missing_ok=True)
                            return result
                        until = next_reset() + 60
                        me.write_text(json.dumps({**mine, "until": until}))
            if until:  # out of the queue order, so the sends behind it go
                wake_waiters()
                print(f"no {entry['problem']} runs left today; waiting for the reset at "
                      f"{datetime.datetime.fromtimestamp(until).isoformat(timespec='minutes')}", flush=True)
                time.sleep(until - time.time())
                me.write_text(json.dumps(mine))
                continue
            if not announced:
                print(f"waiting for a judge slot: place {place + 1} of {len(ahead)} in our queue (cap {CAP} at the judge)",
                      flush=True)
                announced = True
            block_until_change(live[0]["job_id"] if live and len(live) >= CAP else None)
    finally:
        me.unlink(missing_ok=True)
        wake_waiters()


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
    sha = hashlib.sha256(src).hexdigest()
    ran, row = known(sha)
    if not args.force and (row or "submit" in ran):
        sys.exit(f"this source already ran as a scored run, or is pending or waiting ({row or 'ledger'}); "
                 "--force to send it again")
    modes = ("submit",) if "check" in ran and not args.force else ("submit", "check")

    def mode_now():
        left = runs_left(problem)
        return next((m for m in modes if left[m] > 0), None)

    def send(mode):
        # okc reads the file when it sends, so a file edited while this run waited would go out
        # under a receipt holding the old source.
        if args.file.read_bytes() != src:
            sys.exit(f"{args.file} changed while this run waited for a judge slot; not sent. "
                     "Send it again if the new source is what you want.")
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sent_at = time.time()
        result, proc = okc_json(mode, problem, str(args.file), "--no-wait")
        job = pick(result, *JOB_KEYS) if isinstance(result, dict) else None
        job = job or newest_job(problem, mode, sent_at)
        if not job:
            sys.exit(f"okc {mode} failed ({proc.returncode}):\n{proc.stdout}{proc.stderr}")
        start_receipt(problem, mode, job, src, args.file, args.note, stamp)
        return mode, job

    priority = args.priority if args.priority is not None else PRIORITY.get(problem, 3)
    mode, job = send_in_turn({"problem": problem, "file": str(args.file), "sha256": sha}, priority, send, mode_now)
    print(f"okc {mode} {problem} {args.file}: job {job} accepted; waiting for the result "
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
    p = sub.add_parser("send", help="a scored run while the problem has one left today, else a check")
    p.add_argument("file", type=Path)
    p.add_argument("--note", default="", help="what this candidate is, for the ledger")
    p.add_argument("--force", action="store_true", help="send source the ledger already has")
    p.add_argument("--priority", type=float, help="queue order, lower first (default: by problem)")
    p.set_defaults(run=send)
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
