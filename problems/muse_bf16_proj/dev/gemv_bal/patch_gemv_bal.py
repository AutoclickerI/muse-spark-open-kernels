"""Balanced split-K GEMV choices for a decode-layer candidate (small-M projections, T <= 32).

    python3 problems/muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py <cand.py> <out.py> [--all]

`_gb_kernel` is S8's `_gemv_kernel` with runtime k-block ranges: grid (N / BN, S), split s gets k-blocks
[s * KB // S, (s + 1) * KB // S), so S can be any count that keeps every CTA resident in one wave and all
CTAs finish together. Output is the same (P [S, T, N] FP32, S) ABI every consumer reads. A choice
('bal', BN, S, NS) in `_GEMV_CHOICES` routes `gemv()` to it; ROWS maps (tokens, role) to the H200 winners of
prof/lab/gemv_bal.py, applied after `_seed_v9_gemv_choices` (so the prepare tuner still compares the 'custom'
path, now this kernel, against cuBLAS with each role's consumer). --all forces one config on every role and
token count (local screening of the consumers at the new split counts).
"""
import hashlib
import sys
from pathlib import Path

# (tokens, role) -> ('bal', BN, S, NS). H200 26fd09a7 (prof/lab/gemv_bal.py, standalone, cold L2), best vs S8's choice:
# qkv 64/7/4 11.62 / 11.81 / 12.26 us vs 12.51 / 12.77 / 15.34 (cuBLAS, t=32); pool_router 64/6/4 20.94 / 21.09 / 22.05
# vs 23.12 / 22.61 / 25.55; unpooler 64/2/4 18.19 / 18.66 vs cuBLAS 18.85 / 19.01 (t = 1 / 16 / 32). o and
# unpool_slice: no gain (kept). t = 8 takes the t = 1/16 winner (same TB = 16 kernel).
ROWS = {**{(t, 'qkv_gate'): ('bal', 64, 7, 4) for t in (1, 8, 16, 32)},
        **{(t, 'pool_router'): ('bal', 64, 6, 4) for t in (1, 8, 16, 32)},
        **{(t, 'unpooler'): ('bal', 64, 2, 4) for t in (16, 32)}}
ALL = {(t, role): ('bal', 32, 4, 4) for t in (1, 8, 16, 32)
       for role in ('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler')
       if not (role == 'unpooler_slice' and t > 8 or role == 'unpooler' and t <= 8)}

KERNEL = '''

@triton.jit
def _gb_kernel(X, W, P, T, x_stride, N, K, S, BN: tl.constexpr, BK: tl.constexpr, TB: tl.constexpr,
               NS: tl.constexpr, USE_PDL: tl.constexpr):
    """_gemv_kernel with runtime k-block ranges (balanced split-K; P [S, T, N] FP32)."""
    pn = tl.program_id(0)
    ps = tl.program_id(1)
    _pdl_start(USE_PDL)
    n = pn * BN + tl.arange(0, BN)
    kb_total = K // BK
    kb0 = ps * kb_total // S
    kb1 = (ps + 1) * kb_total // S
    _pdl_wait(USE_PDL)
    t = tl.arange(0, TB)
    kk = tl.arange(0, BK)
    acc = tl.zeros((BN, TB), tl.float32)
    for kb in tl.range(kb0, kb1, num_stages=NS):
        k = kb * BK
        w = tl.load(W + n[:, None] * K + (k + kk)[None, :])
        x = tl.load(X + t[:, None] * x_stride + (k + kk)[None, :], mask=t[:, None] < T, other=0.0)
        acc = tl.dot(w, tl.trans(x), acc)
    tl.store(P + ps * (T * N) + t[None, :] * N + n[:, None], acc, mask=t[None, :] < T)


def _gb_gemv(x, w, choice):
    _, BN, S, NS = choice
    T, K = x.shape
    N = w.shape[0]
    P = torch.empty((S, T, N), device=x.device, dtype=torch.float32)
    _launch(_gb_kernel, (N // BN, S), x, w, P, T, x.stride(0), N, K, S, BN=BN, BK=128,
            TB=max(16, triton.next_power_of_2(T)), NS=NS, num_warps=4)
    return P, S


# P7 balanced GEMV (problems/muse_bf16_proj/dev/gemv_bal): (tokens, role) -> ('bal', BN, S, NS)
GB_ROWS = __ROWS__
'''

OLD_CHOICE = '    choice = _GEMV_CHOICES.get((T, N, K))\n'
NEW_CHOICE = OLD_CHOICE + "    if isinstance(choice, tuple) and choice[0] == 'bal':\n        return _gb_gemv(x, w, choice)\n"
OLD_SEED = '            _GEMV_CHOICES[(t, n, k)] = choice\n'
NEW_SEED = OLD_SEED + ('    for (t, name), choice in GB_ROWS.items():\n'
                       '        n, k = state[name].shape\n'
                       '        _GEMV_CHOICES[(t, n, k)] = choice\n')


def _sub(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'expected one {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def patch(src: str, rows=None) -> str:
    rows = ROWS if rows is None else rows
    anchor = src.index('\n\n# (N, K) -> (BN, KS, BK)')  # after the candidate's _gemv_kernel
    src = src[:anchor] + KERNEL.replace('__ROWS__', repr(rows)) + src[anchor:]
    src = _sub(src, OLD_CHOICE, NEW_CHOICE)
    return _sub(src, OLD_SEED, NEW_SEED)


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text(), ALL if '--all' in sys.argv[3:] else None)
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
