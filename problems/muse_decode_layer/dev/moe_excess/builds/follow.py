"""Finite event-driven follow-through for the two P3b jobs.

Read the diagnostic, conservatively remove refuted arms only while the paired
snapshot is still local, then read the one admitted production cohort. No
automatic promotion/scoring, periodic polling, detached model, or shared-doc edit.
"""
import ctypes
import hashlib
import fcntl
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
import candidates as build

H, R = build.H, build.R
DECISION = 'P3b: S22 production builds for large-M MoE ladder outcomes'
sys.path.insert(0, str(R / 'tools'))


def receipt(identity):
    for path in (R / 'runs/muse_moe_w4a16').glob('*.json'):
        try:
            data = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        source = path.with_suffix('.py')
        if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == identity and data.get('status') == 'done':
            return path
    return None


def wait(queue):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC)
    assert fd >= 0
    wd = libc.inotify_add_watch(fd, os.fsencode(R / 'runs/muse_moe_w4a16'), 0x8 | 0x80)
    assert wd >= 0
    assert os.uname().machine == 'x86_64'
    pidfd = libc.syscall(434, queue['pid'], 0)
    if pidfd < 0:
        os.close(fd)
        result = receipt(queue['wrapper_sha256'])
        assert result, 'Sender disappeared without a complete receipt'
        return result
    try:
        while not (result := receipt(queue['wrapper_sha256'])):
            ready, _, _ = select.select([fd, pidfd], [], [])
            if fd in ready:
                os.read(fd, 65536)
            if pidfd in ready:
                result = receipt(queue['wrapper_sha256'])
                assert result, 'Sender completed without matching receipt'
                return result
        return result
    finally:
        os.close(fd)
        os.close(pidfd)


def refuted(raw, summary):
    """A non-largest positive term is still plausible; do not drop it.

    A nonpositive contrast at both Ms refutes its isolated hypothesis only when
    allocation/clone controls overlap one and the relevant cuts keep residency
    with no extra local-memory traffic. Tails/finalizer remain until measured.
    """
    mapping = dict(DECODE=('decode/scale | loaded operands', ('L1', 'L0')),
                   REG=('load issue/reuse/residency | unique stream', ('L0', 'STREAM')),
                   OCC=('load issue/reuse/residency | unique stream', ('L0', 'STREAM')))
    dropped = {}
    for name, (term, cuts) in mapping.items():
        qualified = True
        for m in (128, 256):
            if summary['ranking'][str(m)][term]['CI95'][1] > 0:
                qualified = False
            for phase in ('w13', 'w2'):
                label = f'{m}:{phase}'
                for control in ('allocation', 'clone'):
                    lo, hi = summary['controls'][label][control]['CI95']
                    qualified &= lo <= 1 <= hi
                for arm in cuts:
                    for row in raw['resources'][label + ':' + arm]['info']:
                        qualified &= row[3] == row[4] and row[6] == 0
        if qualified:
            dropped[name] = 'Both M contrasts have CI upper<=0, qualified matched resource/clone controls'
    # PAD cannot help its stated hypothesis if no observed heavy expert tail
    # has a width that changes; do not equate hidden MMA time with zero work.
    tails = [n % 64 for _, _, counts in raw['work_counts'] for n in counts if n > 16]
    if tails and not any(0 < n <= 16 or 32 < n <= 48 for n in tails):
        dropped['PAD'] = 'No sampled heavy expert tail uses a narrower16/48-row tile'
    return dropped


def save_queue():
    manifest = json.loads((H / 'manifest.json').read_text())
    entries = [json.loads(p.read_text()) for p in (R / 'runs/waiting').glob('*.json')]
    row = next((e for e in entries if e.get('sha256') == manifest['wrapper_sha256']), None)
    assert row, 'Paired send is not locally waiting; inspect pending receipts'
    record = dict(pid=row['pid'], wrapper_sha256=row['sha256'],
                  payload_sha256=manifest['payload_sha256'], host=row['problem'],
                  lane=row['lane'], priority=row['priority'], decision=row['decision'],
                  file=row['file'], snapshot='runs/sent/' + row['sha256'][:16] + '/lab_p3b_s22_builds.py',
                  detached_log='runs/detached/' + row['sha256'][:16] + '.log',
                  diagnostic_wrapper_sha256=json.loads((H.parent / 'queue.json').read_text())['wrapper_sha256'],
                  status='queued after diagnostic; native production transfer unmeasured')
    (H / 'queue.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def replace_if_waiting(dropped):
    if not dropped:
        return 'No hypothesis refuted by the conservative predeclared rule'
    queued = json.loads((H / 'queue.json').read_text())
    libc = ctypes.CDLL(None, use_errno=True)
    with (R / 'runs/.dispatch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        row = R / 'runs/waiting' / (str(queued['pid']) + '.json')
        if not row.exists() or json.loads(row.read_text()).get('sha256') != queued['wrapper_sha256']:
            return 'Paired snapshot already admitted; retain original arms and interpret diagnostic afterward'
        pidfd = libc.syscall(434, queued['pid'], 0)
        assert pidfd >= 0
        os.kill(queued['pid'], signal.SIGTERM)
        select.select([pidfd], [], [])
        os.close(pidfd)
    # Only our own local waiter was stopped, under the same dispatch lock used
    # by okc_run. No accepted judge job or another lane can be replaced here.
    current = json.loads((H / 'manifest.json').read_text())
    (H / 'full_manifest.json').write_text(json.dumps(current, indent=2) + '\n')
    enabled = [name for name in current['arms'] if name != 'INC' and name not in dropped]
    builder = build.load(H / 'build_lab.py', 'p3b_night_rebuild')
    builder.main(enabled)
    subprocess.run([sys.executable, 'tools/local_eval.py', str(H / 'out/p3b_s22_builds.py'),
                    '--payload', '--tier', 'static'], cwd=R, check=True)
    subprocess.run([sys.executable, 'tools/local_eval.py', str(H / 'out/wrapper.py'),
                    '--tier', 'wrapper', '--transport', 'lab-only'], cwd=R, check=True)
    subprocess.run([sys.executable, 'tools/h200_lab.py', 'send', str(H / 'out/p3b_s22_builds.py'),
                    '--host', 'muse_moe_w4a16', '--lab-only', '--import-budget', '850', '--check',
                    '--lane', 'P3b', '--decision', DECISION,
                    '--note', 'Same S22 candidates/protocol; pre-admission diagnostic removes only qualified refuted arms: ' + ','.join(dropped)],
                   cwd=R, check=True)
    save_queue()
    return 'Replaced only the local waiter; retained arms ' + ','.join(enabled)


def main():
    parent = json.loads((H.parent / 'queue.json').read_text())
    print('WAIT_DIAGNOSTIC', parent['wrapper_sha256'], flush=True)
    diagnostic = wait(parent)
    subprocess.run([sys.executable, str(H.parent / 'read.py'), str(diagnostic)], cwd=R, check=True)
    reader = build.load(H.parent / 'read.py', 'p3b_diagnostic_reader')
    raw, data = reader.read(diagnostic)
    summary = reader.summarize(raw, data)
    dropped = refuted(raw, summary)
    action = replace_if_waiting(dropped)
    (H / 'diagnostic_selection.json').write_text(json.dumps(dict(receipt=str(diagnostic.relative_to(R)),
        dropped=dropped, action=action, rule='Conservative predeclared outcome/refutation rule in follow.py; no performance sampling or confirmation changes'), indent=2) + '\n')
    print('DIAGNOSTIC', action, flush=True)
    paired = wait(json.loads((H / 'queue.json').read_text()))
    report = build.load(H / 'read.py', 'p3b_production_reader')
    report.analyze(paired)
    print('READ_PRODUCTION', paired, flush=True)


if __name__ == '__main__':
    main()
