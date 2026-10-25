"""Boundary glue at t >= 256 (verify 32): load the weight vectors after griddepcontrol.wait.

    python3 problems/muse_bf16_proj/dev/glue_big/patch_glue_late.py <cand.py> <out.py>

`_boundary_late` is `_boundary_fast` (one CTA per token row, REPLICAS 1) with alpha / beta / prew / postw loaded
where they are used instead of before the wait, so fewer registers stay live; same operations in the same order.
`_best7_boundary` routes a (warps, 1) geometry row to it when the call has more than 128 tokens.

H200 3345699f (prof/lab/glue_big.py: producer F.linear -> boundary, paired vs S10's (16, 1) / (8, 1) / (16, 1),
lab_cold_l2): 16 warps -1.07 / -1.13 us at t = 256 b1 / b2, +0.01 / +0.04 at t = 128 (left on _boundary_fast).
Two kernels over column chunks (stats + apply, C = 4-32) lose +1.5 to +12.8 us, as cluster glue does (1365a290).
"""
import hashlib
import sys
from pathlib import Path

KERNEL = '''

@triton.jit
def _boundary_late(BR, S: tl.constexpr, BS: tl.constexpr, RS: tl.constexpr, STREAM, SOUT, OUT, POSTW, ALPHA, BETA, PREW,
                   NB: tl.constexpr, HAS_POSTW: tl.constexpr, USE_PDL: tl.constexpr):
    """_boundary_fast (REPLICAS 1) with the weight vectors loaded after the wait, where they are used."""
    t = tl.program_id(0)
    _pdl_start(USE_PDL)
    _pdl_wait(USE_PDL)
    j = tl.arange(0, 8192)
    st = tl.load(STREAM + t * 8192 + j)
    br = tl.full((8192,), 0., tl.float32)
    for s in tl.static_range(S):
        br += tl.load(BR + s * BS + t * RS + j, j < NB, 0.).to(tl.float32)
    if HAS_POSTW:
        br *= tl.load(POSTW + j)
    br *= tl.rsqrt(tl.sum(br * br, 0) / 8192 + 1.e-8)
    st = tl.load(ALPHA + j) * st + tl.load(BETA + j) * br
    inv = tl.rsqrt(tl.sum(st * st, 0) / 8192 + 1.e-5)
    tl.store(SOUT + t * 8192 + j, st)
    tl.store(OUT + t * 8192 + j, (st * inv * tl.load(PREW + j)).to(tl.bfloat16))
'''

ANCHOR = '\n\n@triton.jit\ndef _route_fast('
OLD = '''    _launch(_boundary_fast,(t,replicas),p,s,p.stride(0),p.stride(1),stream,sout,out,
        postw if post else alpha,alpha,beta,prew,NB=p.shape[2],HAS_POSTW=post,REPLICAS=replicas,num_warps=warps,use_pdl=True)
'''
NEW = '''    if t>128 and replicas==1:
        _launch(_boundary_late,(t,),p,s,p.stride(0),p.stride(1),stream,sout,out,
            postw if post else alpha,alpha,beta,prew,NB=p.shape[2],HAS_POSTW=post,num_warps=warps,use_pdl=True)
    else:
        _launch(_boundary_fast,(t,replicas),p,s,p.stride(0),p.stride(1),stream,sout,out,
            postw if post else alpha,alpha,beta,prew,NB=p.shape[2],HAS_POSTW=post,REPLICAS=replicas,num_warps=warps,use_pdl=True)
'''


def _sub(src, old, new):
    if src.count(old) != 1:
        raise SystemExit(f'expected one {old[:60]!r}, found {src.count(old)}')
    return src.replace(old, new)


def patch(src: str) -> str:
    src = _sub(src, ANCHOR, KERNEL + ANCHOR)
    return _sub(src, OLD, NEW)


if __name__ == '__main__':
    out = patch(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2], hashlib.sha256(out.encode()).hexdigest())
