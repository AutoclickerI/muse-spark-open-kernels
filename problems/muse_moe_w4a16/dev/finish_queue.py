"""Finish this finite MoE queue, promote passing improvements, and score them.
Only receipt-verified sources are promoted. No new checks or kernel experiments.
Waits on process completion events, without polling timers. Run outside submission code.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
PROBLEM = ROOT / 'problems/muse_moe_w4a16'
DEV = PROBLEM / 'dev'
OKC = str(Path.home() / '.okc/bin/okc')
SUBMISSION = PROBLEM / 'submission.py'
LIBC = ctypes.CDLL(None, use_errno=True)
POLLER = select.poll()
WATCHED = {}
OWNED = []
NEW_SCORES = 0
MAX_NEW_SCORES = 7
BOARD = json.loads((DEV / 'leaderboard_latest.json').read_text())
ACTIONS = []
# These four wrappers started before okc_run.py gained immediate pending receipts.
LEGACY = [
    ('d8f4732ad7bf4d85', 'cand_humming7.py', 'check'),
    ('3d1d9c45cd054c2b', 'cand_humming8.py', 'check'),
    ('f95bdf5e394e474e', 'cand_v7.py', 'check'),
    ('f0eca2fe003f456d', 'cand_v4.py', 'submit'),
]


def emit(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)
    ACTIONS.append(message)


def ledger():
    answer = []
    for line in (ROOT / 'runs/ledger.jsonl').read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get('problem') == 'muse_moe_w4a16':
            answer.append(row)
    return answer


def pending():
    answer = []
    for path in (ROOT / 'runs/pending').glob('*.json'):
        try:
            row = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if row.get('problem') == 'muse_moe_w4a16':
            answer.append(row)
    known = {row['job'] for row in answer + ledger()}
    for job, filename, mode in LEGACY:
        if job not in known:
            source = PROBLEM / filename
            answer.append({'job': job, 'file': str(source.relative_to(ROOT)),
                           'source': str(source.relative_to(ROOT)), 'mode': mode,
                           'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                           'legacy_waiter': True})
    return answer


def result(row):
    return json.loads((ROOT / Path(row['source']).with_suffix('.json')).read_text()).get('result') or {}


def passed(row):
    data = result(row)
    return data.get('passed') is True and not data.get('flags') and isinstance(data.get('speedup'), (int, float))


def eligible(row):
    return row['file'].startswith('problems/muse_moe_w4a16/') and not row.get('source_uncertain') and passed(row)


def watch(pid):
    if pid in WATCHED.values():
        return
    fd = LIBC.syscall(434, pid, 0)
    if fd < 0:
        if ctypes.get_errno() == 3:
            return
        raise OSError(ctypes.get_errno(), 'pidfd_open')
    WATCHED[fd] = pid
    POLLER.register(fd, select.POLLIN)


def discover():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = proc.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(args) >= 4 and args[1].endswith(b'/okc_run.py') and b'/muse_moe_w4a16/' in args[3]:
            watch(int(proc.name))


def launch(args, log):
    with log.open('w') as stream:
        proc = subprocess.Popen([sys.executable, str(ROOT / 'tools/okc_run.py'), *args],
                                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)
    OWNED.append(proc)
    watch(proc.pid)
    emit(f"Started wrapper PID {proc.pid}: {' '.join(args)}; log {log.name}")


def api(*args):
    return json.loads(subprocess.check_output([OKC, *args, '--json'], text=True))


def name(row, rows):
    for other in rows:
        stem = Path(other['file']).stem
        if other['sha256'] == row['sha256'] and stem.startswith('cand_'):
            return stem.removeprefix('cand_')
    return Path(row['file']).stem


def best_row(rows):
    return max((r for r in rows if eligible(r)), key=lambda r: result(r)['speedup'])


def update_report(rows, queued, best, finished=False):
    b = result(best)
    best_name = name(best, rows)
    current_sha = hashlib.sha256(SUBMISSION.read_bytes()).hexdigest()
    scored = [r for r in rows if r['mode'] == 'submit' and passed(r)]
    best_scored = max(scored, key=lambda r: result(r)['speedup']) if scored else None
    you = BOARD.get('you') or {}
    lines = [
        '# Muse NVFP4 MoE results', '',
        'Updated ' + datetime.now(timezone.utc).isoformat(timespec='seconds') + '.', '',
        f"`submission.py` is **{best_name}**, exact receipt source `{best['job']}`: "
        f"**{b['speedup']:.4f}x** H200 geometric mean, CI {b.get('speedup_ci')}. "
        f"Source SHA-256 `{current_sha}`.", '',
        f"Latest verified board: rank **{you.get('rank', '?')}/{BOARD.get('ranked', '?')}**, "
        f"{you.get('speedup', '?')}x. " +
        (f"Best completed scored job `{best_scored['job']}`." if best_scored else 'No passing scored receipt.'), '',
        ('Queue finished.' if finished else f"{len(queued)} remote jobs remain pending; "
         'the finite `dev/finish_queue.py` worker promotes receipt-verified improvements and scores them within quota.'), '',
        '## H200 receipts', '',
        'Ratios are relative to paired SGLang timings. Load columns correspond to batches '
        '**1 / 16 / 32 / 8 / 128 / 256**. Checks do not enter the board. '
        'These are judge measurements; local sm_120 timing is not used as an H200 claim.', '',
        '| Candidate | Job | Mode | GM | D1 | D16 | D32 | V1 | V16 | V32 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        data = result(row)
        ratios = {z['case']['bs']: z['speedup'] for z in data.get('records', [])
                  if z.get('event') == 'benchmark' and 'speedup' in z}
        values = [data.get('speedup')] + [ratios.get(bs) for bs in (1,16,32,8,128,256)]
        cells = [f'{value:.3f}' if isinstance(value, (int,float)) else '-' for value in values]
        lines.append('| ' + ' | '.join([name(row, rows), '`'+row['job']+'`', row['mode'], *cells]) + ' |')
        if not passed(row):
            lines.append(f"\nJob `{row['job']}` did not produce a clean passing result: "
                         f"{str(data.get('error') or data.get('flags') or data.get('status'))[:400]}\n")
    if queued:
        lines += ['', '## Pending jobs', '', '| Candidate | Job | Mode |', '|---|---|---|']
        for row in queued:
            lines.append(f"| {name(row, rows)} | `{row['job']}` | {row['mode']} |")
    lines += ['', '## What changed and what limits it', '',
        'V6 uses an occupancy-sized fully resident grid, eight math warps plus a producer warp, '
        '8-token tiles, two asynchronous weight stages, and one fused call for routing, both '
        'GEMMs, SwiGLU, per-expert norm and weighted reduction. V9 removes the producer warp '
        'and makes thread 0 prefetch, freeing the register budget: offline SM90 compilation '
        'uses 124 registers with zero stack/spills. This is compiler evidence, not measured throughput.', '',
        'V6 profiles contain only the fused kernel (100% of GPU time); they cannot identify '
        'its internal limiting stage. Humming3 GEMMs account for about 95–96% at medium/large '
        'loads; its W13 GEMM alone is about 61%. Nsight Compute was unavailable '
        '(`ERR_NVGPUCTRPERM`), so bandwidth/dequant/stall attribution remains an inference.', '',
        'Humming works with the bundled library and its permitted JIT. TMA and producer warp '
        'specialization improved its H200 geometric mean to 1.2696x (`09ba4af4e6bc4791`), '
        'including 1.4236x at batch 256. Hot/cold bucketing, wider tiles, forced MMA and '
        'Stream-K variants are recorded above as their receipts complete.', '',
        'The hybrid uses native V6 through batch 192 and Humming3 above it. It retains two '
        'lossless packed banks: about 0.906 to 1.812 GB per local layer/shard. This is a '
        'material serving-memory cost, and no end-to-end serving gain has been measured.', '',
        '## Verification', '',
        'All remotely checked variants first passed local standard correctness and fresh-input '
        'CUDA-graph tests on sm_120, except Hopper-only paths, which require the judge. '
        'V6 and V9 also passed nearby batches 2/9/65/300, wide group scales, concentrated '
        'all-local routing, repeated fresh graph inputs, and exact-zero all-remote output; '
        'relative L2 was about 0.0025–0.0026. Humming2/5 passed the same stress tests '
        '(about 0.0046–0.0063). Logs and compiles remain in `dev/`; exact remote source and '
        'results remain in `runs/muse_moe_w4a16/`.', '',
        'Upstream sync ran in an isolated `dev/upstream_sync/` mirror to respect the edit '
        'scope. Official TASK/AGENTS/reference/checks/nvfp4/baseline and llms.txt were unchanged. '
        'No commits/pushes or edits to tools, serving, sim or other problems.', '',
        '## API and serving port', '',
        '`prepare(weights) -> state`; `kernel_fn(data, state) -> BF16 [M,4096]`. Inputs: '
        'BF16 hidden states, int32 top-8 global expert ids, FP32 route weights and shape/rank '
        'config. Weights: packed NVFP4 E2M1 codes, group-16 E4M3 scales, FP32 global scales, '
        'raw BF16 zero-centered post-norm weight. Repacking is lossless; every timed call '
        'recomputes routing and activations. Persistent state has weights and synchronization '
        'control only; generation counters carry no input-derived results.', '',
        'SGLang hook: `MuseSpark1xFp4Experts._forward_marlin` / `_marlin_expert_outputs` '
        'in `python/sglang/srt/layers/muse_spark_fp4_moe.py`. Repack the original ModelOpt '
        'checkpoint before Marlin permutation. Retain pre-expert norm and the surrounding '
        'expert collective (the challenge omits all-reduce), select the actual local EP shard, '
        'and give each concurrent stream/graph private counters and scratch. Extend the '
        'epilogue for distinct real-checkpoint gate/up FP32 global scales. Native norm '
        'expects raw w and computes 1+w; Humming prepare constructs the effective weight. '
        'Normalize each expert before applying routing weights and summing.', '',
        'The native global barrier requires full grid residency. For concurrent serving, '
        'enforce that guarantee, e.g. cooperative launch; isolated occupancy sizing alone '
        'has not verified serving concurrency. NVIDIA: '
        '[cooperative launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html), '
        '[graph support](https://developer.nvidia.com/blog/cuda-11-features-revealed/). '
        'Replace/release the old Marlin bank; retaining duplicate layouts across layers '
        'can consume substantial HBM.', '',
        'Decode-layer reuse: pass normalized expert input, ids and scores; adapt already-local '
        'ids and rename `post_expert_norm` to `post_norm_w`; return the tensor consumed by '
        'unpooling. That task supplies only the 32 local experts. Prefill can reuse the '
        'weight/precision code, but needs wider route indices, scalable distributed alignment '
        'and larger grouped/TMA plans: the current single-CTA routing and plans through 512 '
        'are decode-oriented. Serving and whole-layer performance remain unverified.', '',
        '## Next ideas, ranked by expected gain (unmeasured)', '',
        '1. Native WGMMA with 64-row weight tiles and per-expert 8/16/32-token tiles; '
        'reuse exact dequantization across more tokens and keep activation tiles in shared memory. '
        'Largest architectural opportunity; requires a new lossless layout and SM90a code.',
        '2. Use the measured best native/Humming family over broad batch ranges with one '
        'canonical packed bank, removing the hybrid memory duplication.',
        '3. Serving pre-expert-norm fusion and launch dependency improvements, after expert '
        'GEMM throughput; profiles put nearly all large-batch time in expert computation.', '',
    ]
    if ACTIONS:
        lines += ['## Queue completion actions', ''] + ['- '+action for action in ACTIONS]
    report = PROBLEM / 'RESULTS.md'
    temporary = DEV / 'RESULTS.next.md'
    temporary.write_text('\n'.join(lines))
    os.replace(temporary, report)
    (DEV / 'queue_state.json').write_text(json.dumps({
        'finished': finished, 'best': best, 'pending': queued, 'board': BOARD,
        'actions': ACTIONS, 'new_scores': NEW_SCORES,
    }, indent=2))


def reconcile():
    global NEW_SCORES, BOARD
    rows, queued = ledger(), pending()
    best = best_row(rows)
    source = (ROOT / best['source']).read_bytes()
    assert hashlib.sha256(source).hexdigest() == best['sha256']
    if hashlib.sha256(SUBMISSION.read_bytes()).hexdigest() != best['sha256']:
        temporary = DEV / 'submission.next.py'
        temporary.write_bytes(source)
        os.replace(temporary, SUBMISSION)
        emit(f"Promoted {name(best, rows)} from {best['job']} at {result(best)['speedup']:.6f}x")
    known = {r['sha256'] for r in rows + queued if r['mode'] == 'submit'}
    if best['sha256'] not in known:
        limits = next(p for p in api('problems')['problems'] if p['name'] == 'muse_moe_w4a16')['limits']['scored']
        left = min(limits['left'], MAX_NEW_SCORES - NEW_SCORES)
        checks_pending = any(r['mode'] == 'check' for r in queued)
        if left > 1 or (left == 1 and not checks_pending):
            NEW_SCORES += 1
            launch(['submit', str(SUBMISSION.relative_to(ROOT)), '--note',
                    f"Queue finisher: exact clean passing {name(best, rows)} receipt {best['job']}; GM {result(best)['speedup']:.6f}x"],
                   DEV / f'finish_score_{NEW_SCORES}_{best["job"]}.log')
        elif left == 0:
            emit('No scored capacity remains; best source is promoted but not scored.')
    scored_jobs = {r['job'] for r in rows if r['mode'] == 'submit'}
    if scored_jobs != reconcile.scored_jobs:
        BOARD = api('leaderboard', 'muse_moe_w4a16')
        (DEV / 'leaderboard_latest.json').write_text(json.dumps(BOARD, indent=2))
        reconcile.scored_jobs = scored_jobs
    update_report(rows, queued, best)
    return rows, queued, best


reconcile.scored_jobs = set()
if __name__ == '__main__':
    # One worker only. The lock is released by the OS when this finite worker exits.
    import fcntl
    with (DEV / 'finish_queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        discover()
        emit(f'Watching {len(WATCHED)} existing MoE wrapper processes; no new checks will be sent.')
        while True:
            rows, queued, best = reconcile()
            if not WATCHED:
                if not queued:
                    BOARD = api('leaderboard', 'muse_moe_w4a16')
                    (DEV / 'leaderboard_latest.json').write_text(json.dumps(BOARD, indent=2))
                    emit(f"Queue complete. Board: {BOARD.get('you')}")
                    update_report(rows, [], best, finished=True)
                    break
                # Recover receipts only if their original wrappers have all exited.
                for row in queued:
                    args = ['collect', row['job']]
                    if row.get('legacy_waiter'):
                        args = ['adopt', row['job'], row['file'], '--mode', row['mode'],
                                '--note', 'Recover legacy wrapper after exit; original candidate source retained']
                    launch(args, DEV / f'finish_collect_{row["job"]}.log')
            for fd, event in POLLER.poll():
                pid = WATCHED.pop(fd)
                POLLER.unregister(fd)
                os.close(fd)
                emit(f'Wrapper PID {pid} exited; reading completed receipts.')
            for child in OWNED:
                child.poll()
