"""SM90 compile of swapped-QKV WGMMA and all packing layouts."""
import hashlib
import json
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from sglang.srt.layers import muse_prefill_gemm as p

wg=pack=0
for tokens in (753,3012,16384):
    for index in (0,1,2):
        bm,bn,bk,warps,stages=p.PG_TILES[index]
        for stream in (0,1):
            tiles=triton.cdiv(2304,bm)*triton.cdiv(tokens,bn)
            sm=min(132,tiles*(triton.cdiv(8192,bk) if stream else 1))
            constants=dict(M=2304,N=tokens,K=8192,PN=0,MIXED=False,DUAL=False,
                           BM=bm,BN=bn,BK=bk,SMS=sm,STREAM_K=bool(stream))
            sig=dict(X='*bf16',W='*bf16',LO='*bf16',Y='*bf16',R='*bf16',PART='*fp32',COUNT='*i32',
                     **{k:'constexpr' for k in constants})
            kernel=triton.compile(ASTSource(p._pg_matmul,signature=sig,constexprs=constants),
                target=GPUTarget('cuda',90,32),options=dict(num_warps=warps,num_stages=stages))
            assert 'wgmma.mma_async' in kernel.asm['ptx'];wg+=1
    for stride in [(width,1) for width in p.PG_QKV_PAD]+[(1,tokens)]:
        constants=dict(M=tokens,N=2304,S0=stride[0],S1=stride[1],B=32)
        triton.compile(ASTSource(p._pg_pack_matrix,signature=dict(A='*bf16',Y='*bf16',
            **{k:'constexpr' for k in constants}),constexprs=constants),
            target=GPUTarget('cuda',90,32),options=dict(num_warps=4))
        pack+=1
p._pg_native('9.0a')
print(json.dumps(dict(ok=True,swapped_wgmma=wg,packing=pack,native_lt='sm90a built',
                     module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())))
