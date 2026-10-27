"""Wait for this sender's receipt event, then materialize the validated report.

Local helper only. Uses inotify and pidfd completion signals; no polling, judge
calls, extra quota, model invocation, automatic promotion or shared-doc edits.
"""
import ctypes
import hashlib
import json
import os
import select
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def matching_receipt(directory, identity):
    for path in directory.glob('*.json'):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        source = path.with_suffix('.py')
        if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == identity and data.get('status') == 'done':
            return path
    return None


def main():
    queue = json.loads((HERE / 'queue.json').read_text())
    directory = ROOT / 'runs' / queue['host']
    directory.mkdir(exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    # Watch before the initial scan so a receipt cannot fall between them.
    watch = libc.inotify_add_watch(fd, os.fsencode(directory), 0x8 | 0x80)
    if watch < 0:
        raise OSError(ctypes.get_errno(), 'inotify_add_watch')
    # Host Python/glibc omit pidfd_open; Linux x86_64's syscall is 434
    # (/usr/include/x86_64-linux-gnu/asm/unistd_64.h).
    assert os.uname().machine == 'x86_64'
    pidfd = libc.syscall(434, queue['pid'], 0)
    if pidfd < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    print('WAIT', queue['wrapper_sha256'], 'sender', queue['pid'], flush=True)
    try:
        while not (receipt := matching_receipt(directory, queue['wrapper_sha256'])):
            ready, _, _ = select.select([fd, pidfd], [], [])
            if fd in ready:
                os.read(fd, 65536)
            if pidfd in ready and not matching_receipt(directory, queue['wrapper_sha256']):
                raise RuntimeError('sender exited without a matching completed receipt; inspect detached log')
        subprocess.run([sys.executable, str(HERE / 'read.py'), str(receipt)], check=True)
        destination = HERE / 'result' / 'collection.json'
        destination.write_text(json.dumps(dict(status='validated-native-report-ready',
                                               receipt=str(receipt.relative_to(ROOT)),
                                               wrapper_sha256=queue['wrapper_sha256'],
                                               promotion='unmeasured; no candidate'), indent=2) + '\n')
        print('READ', receipt, flush=True)
    finally:
        os.close(pidfd)
        os.close(fd)


if __name__ == '__main__':
    main()
