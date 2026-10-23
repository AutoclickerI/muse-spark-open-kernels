"""Build actual Hopper specializations and confirm dense tensor math uses WGMMA."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from sglang.srt.layers import muse_prefill_gemm as p


def build(case):
    m, n, k, pn, mixed, dual, index, streamk = case
    bm, bn, bk, warps, stages = p.PG_TILES[index]
    tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
    sm = min(132, tiles * (triton.cdiv(k, bk) if streamk else 1))
    constants = dict(M=m, N=n, K=k, PN=pn, MIXED=mixed, DUAL=dual,
                     BM=bm, BN=bn, BK=bk, SMS=sm, STREAM_K=bool(streamk))
    signature = {**{name: '*bf16' for name in ('X', 'W', 'LO')},
                 'Y': '*fp32' if dual else '*bf16', 'R': '*fp32', 'PART': '*fp32', 'COUNT': '*i32',
                 **{name: 'constexpr' for name in constants}}
    code = triton.compile(ASTSource(p._pg_matmul, signature=signature, constexprs=constants),
                           target=GPUTarget('cuda', 90, 32),
                           options=dict(num_warps=warps, num_stages=stages))
    assert 'wgmma.mma_async' in code.asm['ptx'], ('not WGMMA', case)
    return dict(case=case, shared_bytes=code.metadata.shared, wgmma=True)


if __name__ == '__main__':
    cases = [(m, n, k, pn, mixed, dual, index, streamk)
             for m in (753, 3012, 16384)
             for n, k, pn, mixed, dual in ((2304,8192,0,False,False), (8192,1024,0,False,False),
                 (4096,8192,0,False,False), (8192,4096,0,False,False),
                 (256,8192,0,False,True), (4352,8192,4096,True,False))
             for index in range(len(p.PG_TILES)) for streamk in (0,1)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(build, cases))
    pointwise = 0
    for mixed in (False, True):
        for dtype in ('*bf16', '*fp32'):
            constants = dict(PN=4096, RN=256, EPS=1e-5, MIXED_INPUT=mixed, B=4096)
            triton.compile(ASTSource(p._pg_mixed_norm,
                signature=dict(A='*fp32' if mixed else '*bf16', Y='*bf16', R='*fp32', GAIN=dtype,
                               **{name:'constexpr' for name in constants}), constexprs=constants),
                target=GPUTarget('cuda',90,32), options=dict(num_warps=4))
            pointwise += 1
    p._pg_native('9.0a')
    report = dict(ok=True, specializations=len(rows), wgmma=sum(r['wgmma'] for r in rows), norm_epilogues=pointwise,
                  max_shared=max(r['shared_bytes'] for r in rows), native_lt='sm90a built',
                  module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())
    print(json.dumps(report))
