"""Exit on the next wrapper completion. Linux pidfds, no network/status polling."""
import ctypes
import json
import select
from pathlib import Path

repo = Path(__file__).resolve().parents[3]
ledger = repo / "runs/ledger.jsonl"


def rows():
    return [json.loads(line) for line in ledger.read_text().splitlines()
            if '"muse_moe_w4a16"' in line]


initial = rows()
seen = {row["job"] for row in initial}
for row in initial[-2:]:
    print("Latest receipt:", json.dumps({k: row.get(k) for k in ("file", "job", "mode", "speedup")}), flush=True)
poller = select.poll()
pending = {}
libc = ctypes.CDLL(None, use_errno=True)
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        argv = proc.joinpath("cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if len(argv) < 4 or not argv[1].endswith(b"/okc_run.py") or b"/muse_moe_w4a16/" not in argv[3]:
        continue
    fd = libc.syscall(434, int(proc.name), 0)
    if fd < 0:
        if ctypes.get_errno() == 3:
            continue
        raise OSError(ctypes.get_errno(), "pidfd_open")
    pending[fd] = (int(proc.name), argv[2].decode(), argv[3].decode())
    poller.register(fd, select.POLLIN)
print("Waiting on", len(pending), "active MoE wrappers", flush=True)
if pending:
    for fd, _ in poller.poll():
        print("Completed:", pending[fd], flush=True)
    for row in rows():
        if row["job"] not in seen:
            print(json.dumps(row), flush=True)
