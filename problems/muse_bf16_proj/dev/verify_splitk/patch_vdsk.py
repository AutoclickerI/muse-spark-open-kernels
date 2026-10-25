"""Add verify-dense split-K projection variants (M=128/256) to a decode-layer candidate's tuner.

    python3 problems/muse_bf16_proj/dev/verify_splitk/patch_vdsk.py <cand.py> <out.py>

The candidate's prepare-time projection tuner (`_tune_projections`) already times every variant with
its real consumer (qk_prep, boundary, route_and_norm) on the judge's H200 and keeps the fastest, so
these variants are added as options, never forced. A variant ('vdsk', cfg, S) runs a Triton
WGMMA split-K kernel that writes FP32 partials P[S, M, N], the (P, S) ABI those consumers read
(the T<=64 GEMV path uses it). Configs compile in a background thread at prepare start.
"""
import hashlib
import sys
from pathlib import Path

# (tokens, role) -> [((BT, BW, BK, stages, warps, TMA), S)]; H200 lab b15a3bde (with a consumer proxy):
# only M=128 qkv_gate wins (20.42 vs 23.26 us, 0.8777); every other role/M loses 1.03-1.19x to cuBLAS.
SHORTLIST = {
    (128, 'qkv_gate'): [((128, 64, 64, 8, 4, 1), 3)],
}

BLOCK = r'''

# ---- verify-dense split-K projections (P7; problems/muse_bf16_proj/dev/verify_splitk) ----
try:
    from triton.tools.tensor_descriptor import TensorDescriptor as _VDSK_TD
except ImportError:
    _VDSK_TD = None
VDSK_SHORTLIST = __SHORTLIST__


@triton.jit
def _vdsk_gemm(X, W, P, x_desc, w_desc, M, N, K, S,
               BT: tl.constexpr, BW: tl.constexpr, BK: tl.constexpr, NS: tl.constexpr, TMA: tl.constexpr):
    pid = tl.program_id(0)
    ps = tl.program_id(1)
    nw = N // BW
    pw = pid % nw
    pt = pid // nw
    ksteps = K // BK
    k_lo = ps * ksteps // S
    k_hi = (ps + 1) * ksteps // S
    rt = pt * BT + tl.arange(0, BT)
    rw = pw * BW + tl.arange(0, BW)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BT, BW), dtype=tl.float32)
    for kk in tl.range(k_lo, k_hi, num_stages=NS):
        if TMA:
            x = x_desc.load([pt * BT, kk * BK])
            w = w_desc.load([pw * BW, kk * BK])
        else:
            x = tl.load(X + rt[:, None] * K + (kk * BK + rk)[None, :], mask=rt[:, None] < M, other=0.0)
            w = tl.load(W + rw[:, None] * K + (kk * BK + rk)[None, :])
        acc = tl.dot(x, w.T, acc)
    tl.store(P + ps * M * N + rt[:, None] * N + rw[None, :], acc, mask=rt[:, None] < M)


def _vdsk_project(x, w, variant):
    _, cfg, S = variant
    BT, BW, BK, NS, NWARPS, TMA = cfg
    M, K = x.shape
    N = w.shape[0]
    if TMA and _VDSK_TD is None:
        raise RuntimeError('vdsk TMA variant without TensorDescriptor')
    P = torch.empty((S, M, N), device=x.device, dtype=torch.float32)
    xd = _VDSK_TD.from_tensor(x, [BT, BK]) if TMA else None
    wd = _VDSK_TD.from_tensor(w, [BW, BK]) if TMA else None
    _vdsk_gemm[(-(-M // BT) * (N // BW), S)](x, w, P, xd, wd, M, N, K, S, BT=BT, BW=BW, BK=BK, NS=NS, TMA=TMA,
                                             num_warps=NWARPS, num_stages=NS)
    return P, S


def _vdsk_warm(state):
    """Compile every shortlisted config once (background thread at prepare start)."""
    for (t, name), entries in VDSK_SHORTLIST.items():
        w = state[name]
        x = torch.zeros((t, w.shape[1]), device=w.device, dtype=torch.bfloat16)
        for cfg, S in entries:
            try:
                _vdsk_project(x, w, ('vdsk', cfg, S))
            except Exception as exc:
                print(f'VDSK_WARM_REJECT t={t} name={name} cfg={cfg} S={S} {type(exc).__name__}: {str(exc)[:80]}', flush=True)
    torch.cuda.synchronize()
'''


def patch(src: str) -> str:
    anchor_imports = 'import triton.language as tl\n'
    assert src.count(anchor_imports) >= 1
    block = BLOCK.replace('__SHORTLIST__', repr(SHORTLIST))
    i = src.index(anchor_imports) + len(anchor_imports)
    # The block needs torch/triton only; place it right after the first triton.language import.
    src = src[:i] + block + src[i:]
    old_pv = 'def _project_variant(x, w, name, state, variant, out_dtype=torch.bfloat16):\n'
    assert src.count(old_pv) == 1
    src = src.replace(old_pv, old_pv + "    if isinstance(variant,tuple) and variant[0]=='vdsk':return _vdsk_project(x,w,variant)\n")
    old_var = '            variants=["custom", "cublas"]\n'
    assert src.count(old_var) == 1
    src = src.replace(old_var, old_var + "            variants += [('vdsk',cfg,S) for cfg,S in VDSK_SHORTLIST.get((t,name),())]\n")
    # Warm the shortlisted configs while the weights and native modules are prepared.
    old_prep = '        state=_prepare_weights(weights)\n'
    assert src.count(old_prep) == 1
    src = src.replace(old_prep, old_prep + '        vdsk_future=native_pool.submit(_vdsk_warm,state)\n')
    old_join = "        state['moe_nt2_mod']=nt2_future.result()\n"
    assert src.count(old_join) == 1
    src = src.replace(old_join, old_join + '        vdsk_future.result()\n')
    return src


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
