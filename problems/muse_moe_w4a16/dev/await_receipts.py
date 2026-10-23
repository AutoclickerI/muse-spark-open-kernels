"""Wait for running okc wrappers to exit using pidfds; no timed status polling."""
import json
import ctypes
import os
import select
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
poller = select.poll()
pending = {}
for arg in sys.argv[1:]:
    pid = int(arg)
    fd = ctypes.CDLL(None, use_errno=True).syscall(434, pid, 0)
    if fd < 0:
        if ctypes.get_errno() != 3:
            raise OSError(ctypes.get_errno(), "pidfd_open")
        continue
    poller.register(fd, select.POLLIN)
    pending[fd] = pid

seen = set()


def report():
    ledger = repo / "runs/ledger.jsonl"
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row["problem"] != "muse_moe_w4a16" or row["job"] in seen:
            continue
        seen.add(row["job"])
        print(json.dumps({k: row.get(k) for k in ("file", "job", "mode", "status", "speedup")}), flush=True)


report()
while pending:
    for fd, _ in poller.poll():
        print("wrapper completed", pending.pop(fd), flush=True)
        poller.unregister(fd)
        os.close(fd)
        report()
