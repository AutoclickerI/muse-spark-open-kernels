"""Compile all DFlash pointwise specializations for SM90 without executing them."""
import hashlib,json
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from sglang.srt.layers import muse_dflash as d

compiled=[]
def build(fn,sig,const,fp_fusion=False):
    code=triton.compile(ASTSource(fn,signature=sig,constexprs=const),
                        target=GPUTarget('cuda',90,32),
                        options=dict(num_warps=4,enable_fp_fusion=fp_fusion))
    compiled.append(dict(kernel=fn.__name__,const=const,shared_bytes=code.metadata.shared))
for c in (1,4,16,32):
    for dtype in ('*bf16','*fp32'):
        build(d._df_unary_kernel,dict(X=dtype,Y='*fp32',N='constexpr',SCALE='constexpr',CAP='constexpr',B='constexpr'),
              dict(N=7*c*16,SCALE=.1767766952966369,CAP=20.,B=256))
    build(d._df_conv_kernel,{**{s:'*bf16' for s in ('X','D','W','O')},
                            **{s:'constexpr' for s in ('T','H','G','DT','DS','DG','BLOCK','B')}},
          dict(T=c*8,H=8192,G=16,DT=2048,DS=512,DG=1,BLOCK=8,B=256))
    build(d._df_selector_pack_kernel,{**{s:'*bf16' for s in ('A','B','H','P','Q')},
                                     'IDS':'*i64','ANCHOR':'*i64',
                                     **{s:'constexpr' for s in ('C','L','K','R','V','BATCH')}},
          dict(C=c,L=7,K=16,R=256,V=202048,BATCH=256),True)
d._df_proj._module('9.0a')
print(json.dumps(dict(ok=True,sm90_pointwise=len(compiled),sm90a_dense='built',
                     module_sha256=hashlib.sha256(Path(d.__file__).read_bytes()).hexdigest(),kernels=compiled),indent=2))
