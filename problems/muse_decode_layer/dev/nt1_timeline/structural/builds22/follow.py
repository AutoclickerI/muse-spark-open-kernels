"""Finite receipt-driven diagnostic pruning and one paired-cohort handoff.

No periodic polling, automatic scoring, shared-doc writes, or accepted-job
cancellation. Only this lane's still-local paired waiter can be replaced.
"""
import ctypes
import fcntl
import hashlib
import json
import os
import select
import signal
import statistics
import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[5]
sys.path.insert(0, str(R / 'tools'))
import promotion


def read(path):
    return json.loads(path.read_text())


def run(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=R, check=True)


def matching_rows(identity):
    rows = []
    for line in (R / 'runs/ledger.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('sha256') == identity:
            rows.append(row)
    for path in (R / 'runs/pending').glob('*.json'):
        try:
            row = read(path)
        except (ValueError, OSError):
            continue
        if row.get('sha256') == identity:
            rows.append(row)
    return rows


def receipt(identity):
    for row in matching_rows(identity):
        path = (R / row['source']).with_suffix('.json')
        if not path.exists():
            continue
        try:
            data = read(path)
        except (ValueError, OSError):
            continue
        if data.get('status') == 'done' and 'result' in data:
            assert hashlib.sha256(path.with_suffix('.py').read_bytes()).hexdigest() == identity
            return path
    return None


def wait(queue):
    """Actual sender exit or receipt-file events, with a lost-sender recovery."""
    identity = queue['wrapper_sha256']
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    assert fd >= 0
    for path in (R / 'runs/muse_moe_w4a16', R / 'runs/pending'):
        assert libc.inotify_add_watch(fd, os.fsencode(path), 0x8 | 0x80) >= 0
    pidfd = libc.syscall(434, queue['sender_pid'], 0)  # x86_64 pidfd_open
    assert os.uname().machine == 'x86_64'
    try:
        while not (path := receipt(identity)):
            if pidfd < 0:
                pending = matching_rows(identity)
                assert pending, 'Sender exited without a receipt or pending judge job'
                run('tools/h200_lab.py', 'read', pending[-1]['job'])
                path = receipt(identity)
                assert path, 'Read finished without the pinned receipt'
                return path
            ready, _, _ = select.select([fd, pidfd], [], [])
            if fd in ready:
                os.read(fd, 65536)
            if pidfd in ready:
                os.close(pidfd)
                pidfd = -1
        return path
    finally:
        os.close(fd)
        if pidfd >= 0:
            os.close(pidfd)


def refuted(data):
    """Only close an outcome when its upper saving bound is <=0 everywhere.

    A positive non-largest term remains plausible. Oracle terms overlap; their
    positive savings are never added. This cannot close a different recipe.
    """
    assert data['measured'] and data['source_audit']['verified']
    ms = (1, 8, 16, 32)
    def nonpositive(term):
        return all(next(v for v in data['contrasts'] if v['M'] == m and v['term'] == term)['CI95'][1] <= 0 for m in ms)
    dropped = {}
    lazy = [v for v in data['ratios'] if v['arm'] == 'LAZY']
    lazy_loses = len(lazy) == 4 and all(v['measured'] and v['CI95'][0] > 1 for v in lazy)
    if lazy_loses:
        dropped['LAZY'] = 'Correct native LAZY has time-ratio CI lower>1 at every M'
    if lazy_loses and nonpositive('accumulator_recurrence'):
        dropped['LAZY2'] = 'LAZY loses everywhere and matched recurrence oracle saving upper<=0 everywhere'
    if nonpositive('ideal_useful_MMA_count'):
        dropped['N1SIMT'] = 'Matched useful-N instruction oracle saving upper<=0 at every M'
    if nonpositive('epilogue_NOWAIT'):
        dropped['EPI4'] = 'Matched epilogue-NOWAIT increment upper<=0 at every M'
    def bank_nonpositive(m):
        a = next(v for v in data['contrasts'] if v['M'] == m and v['term'] == 'global_to_shared_fragment')['values']
        b = next(v for v in data['contrasts'] if v['M'] == m and v['term'] == 'cached_register_lifetime')['values']
        values = [x+y for x,y in zip(a,b)]  # exact BASE_PAD minus FRAG_JIT, seed by seed
        return statistics.mean(values)+promotion.T95[7]*statistics.stdev(values)/(8**.5) <= 0
    if nonpositive('pack_free_register') and all(bank_nonpositive(m) for m in ms):
        dropped['CACHE1'] = 'Both register and just-in-time shared-fragment saving upper<=0 at every M'
    return dropped


def replace_if_local(dropped):
    if not dropped:
        return 'No outcome refuted by the predeclared conservative rule'
    q = read(D / 'queue.json')
    libc = ctypes.CDLL(None, use_errno=True)
    with (R / 'runs/.dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = R / q['waiting']
        if not path.exists():
            return 'Paired job already admitted; original arms retained, no cancellation'
        waiter = read(path)
        assert waiter['sha256'] == q['wrapper_sha256'] and waiter['lane'] == 'P3a'
        args = (Path('/proc') / str(q['sender_pid']) / 'cmdline').read_bytes().split(b'\0')
        assert q['expected_source'].encode() in args and any(v.endswith(b'tools/okc_run.py') for v in args)
        pidfd = libc.syscall(434, q['sender_pid'], 0)
        assert pidfd >= 0
        os.kill(q['sender_pid'], signal.SIGTERM)
        select.select([pidfd], [], [])
        os.close(pidfd)
        assert not path.exists()
    (D / 'unpruned_queue.json').write_text(json.dumps(q, indent=2)+'\n')
    (D / 'unpruned_lab_manifest.json').write_text((D / 'lab_manifest.json').read_text())
    active = [name for name in q['active'] if name not in dropped]
    (D / 'active.json').write_text(json.dumps(active, indent=2)+'\n')
    if not active:
        q.update(state='All predeclared outcome arms refuted before admission; no paired quota spent', active=[])
        (D / 'queue.json').write_text(json.dumps(q, indent=2)+'\n')
        return q['state']
    # Reconstruct pinned ignored source artifacts if they have been cleaned.
    run(D / 'build.py')
    run(D / 'build_lab.py')
    run('tools/local_eval.py', 'prof/lab/moe_p3a_s22.py', '--payload', '--tier', 'static')
    run('tools/local_eval.py', 'runs/screen/p3a_builds22/wrapper.py', '--tier', 'wrapper', '--transport', 'lab-only')
    from send_batch import send
    send(replace=True)
    return 'Replaced only the unsubmitted waiter; retained '+','.join(active)


def main():
    seal = read(D / 'follow_seal.json')
    assert read(D / 'queue.json')['wrapper_sha256'] == seal['paired_wrapper_sha256']
    for rel, digest in seal['source_sha256'].items():
        assert hashlib.sha256((R / rel).read_bytes()).hexdigest() == digest, 'Follow-through source changed: '+rel
    diagnostic = wait(read(D.parent / 'operand/queue.json'))
    selection = dict(receipt=str(diagnostic.relative_to(R)), dropped={}, diagnostic_measured=False)
    try:
        run(D.parent / 'operand/read.py', diagnostic)
        run(D.parent / 'operand/report.py')
        data = read(D.parent / 'operand/results.json')
        dropped = refuted(data)
        selection.update(diagnostic_measured=True, dropped=dropped)
    except Exception as error:
        selection.update(action='Diagnostic failed audit; retain every correct candidate, no recipe closed', error=repr(error))
    if selection['diagnostic_measured']:
        try:
            selection['action'] = replace_if_local(selection['dropped'])
        except Exception as error:
            selection.update(action='Pruning/rebuild failed; inspect the owned queue before any resend', error=repr(error))
            (D / 'diagnostic_selection.json').write_text(json.dumps(selection, indent=2)+'\n')
            raise
    (D / 'diagnostic_selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    print('DIAGNOSTIC_SELECTION', selection, flush=True)
    if not read(D / 'queue.json')['active']:
        run(D / 'report.py')
        return
    paired = wait(read(D / 'queue.json'))
    run(D / 'read.py', paired)
    q = read(D / 'queue.json')
    q.update(job=read(paired)['id'], receipt=str(paired.relative_to(R)), state='Complete; see results and P4_HANDOFF for per-arm measurement validity')
    (D / 'queue.json').write_text(json.dumps(q, indent=2)+'\n')
    run(D / 'report.py')
    run(D.parents[1] / 'review/rejudge.py')
    print('PAIRED_HANDOFF', paired, flush=True)


if __name__ == '__main__':
    main()
