#!/usr/bin/env python3
"""Send a file to the judge with `okc check` or `okc submit`, and keep a receipt.

    tools/okc_run.py check  problems/muse_moe_w4a16/submission.py --note "split-K w2"
    tools/okc_run.py submit problems/muse_moe_w4a16/submission.py --note "..."

Each run leaves, under runs/<problem>/, <utc>_<mode>_<job>.py (the exact source sent) and
<utc>_<mode>_<job>.json (okc's whole result), and one line in runs/ledger.jsonl. okc itself
also writes profiles/<file>.<job>.txt and .log beside the file.

Runs are capped per problem per day (10 scored, 20 checks; reset 00:00 UTC), and the judge
answers the same source again with its earlier result, so this refuses source the ledger
already has for the same mode unless --force. Host Python, standard library only.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEDGER = REPO / "runs" / "ledger.jsonl"


def okc_bin() -> str:
    return shutil.which("okc") or str(Path.home() / ".okc" / "bin" / "okc")


def okc_json(*args: str) -> dict:
    proc = subprocess.run([okc_bin(), *args, "--json"], capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.exit(f"okc {' '.join(args)} failed ({proc.returncode}):\n{proc.stdout}{proc.stderr}")


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["check", "submit"])
    ap.add_argument("file", type=Path)
    ap.add_argument("--note", default="", help="what this candidate is, for the ledger")
    ap.add_argument("--force", action="store_true", help="send source the ledger already has")
    args = ap.parse_args()

    src = args.file.read_bytes()
    m = re.match(rb"#\s*okc:\s*problem=(\w+)", src)
    if not m:
        sys.exit(f"{args.file}: the first line must be '# okc: problem=<name>'")
    problem = m.group(1).decode()
    sha = hashlib.sha256(src).hexdigest()
    if LEDGER.exists() and not args.force:
        for line in LEDGER.read_text().splitlines():
            row = json.loads(line)
            if row["sha256"] == sha and row["mode"] == args.mode:
                sys.exit(f"this source already ran as {args.mode} job {row['job']} ({row['status']}, "
                         f"speedup {row['speedup']}); --force to send it again")

    sent = okc_json(args.mode, problem, str(args.file), "--no-wait")
    job = pick(sent, "job_id", "job", "id")
    if not job:
        sys.exit(f"okc returned no job id: {sent}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "runs" / problem
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{stamp}_{args.mode}_{job}"
    stem.with_suffix(".py").write_bytes(src)
    print(f"{args.mode} job {job} sent; waiting (receipt {stem.relative_to(REPO)}.*)", flush=True)

    # okc status writes the profile and log beside the file only from the file's directory.
    result = okc_json("status", job, "--wait")
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1, sort_keys=True))
    row = {
        "utc": stamp, "problem": problem, "mode": args.mode, "job": job, "sha256": sha,
        "file": os.path.relpath(args.file.resolve(), REPO), "source": str(stem.with_suffix(".py").relative_to(REPO)),
        "git": git_rev(), "status": pick(result, "status"), "speedup": pick(result, "speedup"),
        "interval": pick(result, "interval", "speedup_interval", "ci"), "note": args.note,
    }
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=1))


if __name__ == "__main__":
    main()
