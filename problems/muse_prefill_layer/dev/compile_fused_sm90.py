"""Compile all new fused grouped BF16/direct expert algorithms for SM90."""
from pathlib import Path
import importlib.util,json
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=Path('problems/muse_prefill_layer/cand_fused_expert_portfolio.py')
s=importlib.util.spec_from_file_location('fused_compile',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for name,(direct,bm,fn,bk,warps,stages,split) in m._FUSED_EXPERT_CHOICES.items():
 for gated in (True,False):
  sig=dict(A='*bf16',W='*u8' if direct else '*bf16',SF='*u8',OUT='*bf16',PART='*fp32' if split>1 else '*bf16',STARTS='*i32',COUNTS='*i32',TILES='*i32',GS='*fp32',GS_UP='*fp32',CAP='i32',GATED='constexpr',DIRECT='constexpr',BM='constexpr',FN='constexpr',BK='constexpr',SPLIT='constexpr',GRID='constexpr')
  c=triton.compile(ASTSource(m._fused_grouped_gemm,signature=sig,constexprs=dict(GATED=gated,DIRECT=direct,BM=bm,FN=fn,BK=bk,SPLIT=split,GRID=264)),target=GPUTarget('cuda',90,32),options=dict(num_warps=warps,num_stages=stages))
  item=dict(choice=name,gated=gated,metadata=str(c.metadata));rows.append(item);print(item,flush=True)
  (p.parent/'dev'/f'fused_{name}_{gated}_sm90.ptx').write_text(c.asm['ptx'])
for split in (2,4):
 for gated in (True,False):
  sig=dict(PART='*fp32',OUT='*bf16',STARTS='*i32',COUNTS='*i32',GS='*fp32',GS_UP='*fp32',CAP='i32',GATED='constexpr',SPLIT='constexpr',BR='constexpr')
  c=triton.compile(ASTSource(m._fused_split_epilogue,signature=sig,constexprs=dict(GATED=gated,SPLIT=split,BR=1)),target=GPUTarget('cuda',90,32),options=dict(num_warps=4))
  rows.append(dict(split=split,gated=gated,metadata=str(c.metadata)));print('SM90 split epilogue PASS',split,gated,flush=True)
for bm in (64,128):
 c=triton.compile(ASTSource(m._fused_plan,signature=dict(COUNTS='*i32',TILES='*i32',BM='constexpr'),constexprs=dict(BM=bm)),target=GPUTarget('cuda',90,32),options=dict(num_warps=1))
 rows.append(dict(plan_bm=bm,metadata=str(c.metadata)))
c=triton.compile(ASTSource(m._fused_flush_read,signature=dict(X='*u8',OUT='*i32',N='constexpr',B='constexpr',GRID='constexpr'),constexprs=dict(N=1<<30,B=4096,GRID=1056)),target=GPUTarget('cuda',90,32),options=dict(num_warps=4))
rows.append(dict(flush=True,metadata=str(c.metadata)))
(p.parent/'dev/fused_sm90.json').write_text(json.dumps(rows,indent=2))
print('SM90 fused expert compile PASS',len(rows),'specializations',flush=True)
