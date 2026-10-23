"""Compile every persistent dense specialization for the judge's SM90 target."""
from pathlib import Path
import importlib.util,json
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=Path('problems/muse_prefill_layer/cand_score_portfolio.py')
s=importlib.util.spec_from_file_location('score_triton_compile',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for name,(n,k) in dict(qkv=(2304,8192),o=(8192,1024),pooler=(4096,8192),unpooler=(8192,4096),router_cat=(512,8192)).items():
 for config,(bm,bn,bk,w,st) in m._DENSE_TRITON.items():
  sig=dict(A='*bf16',B='*bf16',C='*fp32' if name=='router_cat' else '*bf16',M='i32',N='constexpr',K='constexpr',BM='constexpr',BN='constexpr',BK='constexpr',GRID='constexpr')
  const=dict(N=n,K=k,BM=bm,BN=bn,BK=bk,GRID=132)
  c=triton.compile(ASTSource(m._dense_persistent,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options=dict(num_warps=w,num_stages=st))
  row=dict(projection=name,choice=config,registers=c.metadata.n_regs if hasattr(c.metadata,'n_regs') else None,metadata=str(c.metadata))
  rows.append(row);print(row,flush=True)
  (p.parent/'dev'/f'score_dense_{name}_{config}_sm90.ptx').write_text(c.asm['ptx'])
(p.parent/'dev/score_dense_sm90.json').write_text(json.dumps(rows,indent=2))
print('SM90 all 15 dense persistent specializations PASS',flush=True)
