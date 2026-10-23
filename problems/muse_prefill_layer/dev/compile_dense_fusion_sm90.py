"""Cross-compile every new dense specialization for Hopper without executing H200 kernels."""
from pathlib import Path
import importlib.util,json
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
p=Path('problems/muse_prefill_layer/cand_dense_fusion_portfolio.py')
s=importlib.util.spec_from_file_location('dense_compile',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
def comp(fn,types,const,warps=4,stages=2):
 sig={n:('constexpr' if n in const else types[n]) for n in fn.arg_names}
 c=triton.compile(ASTSource(fn,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options=dict(num_warps=warps,num_stages=stages))
 rows.append(dict(fn=fn.__name__,const=const,metadata=str(c.metadata)));print('SM90 PASS',fn.__name__,const,flush=True)
ptr=dict(A='*bf16',W='*bf16',P='*bf16',L='*fp32',BIAS='*fp32',LS='*i32',IDS='*i32',PROBS='*fp32',E='*i32',POS='*i32',COUNTS='*i32',T='i32')
for route in (False,True):
 comp(m._dense_merged_persistent,ptr,dict(BM=64,BN=128,BK=64,GRID=132,ROUTE=route,RBM=16))
 comp(m._dense_merged_split,dict(Z='*fp32',P='*bf16',LOGITS='*fp32',BIAS='*fp32',LS='*i32',IDS='*i32',PROBS='*fp32',E='*i32',POS='*i32',COUNTS='*i32',T='i32'),dict(BM=1,ROUTE=route))
for K in (1024,4096):
 comp(m._dense_streamk_balanced,dict(A='*bf16',W='*bf16',PART='*fp32',T='i32'),dict(K=K,GRID=264))
 for T in (753,3012,16384):
  work=triton.cdiv(triton.cdiv(T,64)*64*(K//64),264)
  comp(m._dense_streamk_boundary,dict(PART='*fp32',STREAM='*fp32',NEW='*fp32',OUT='*bf16',POST='*fp32',ALPHA='*fp32',BETA='*fp32',PRE='*fp32',T='i32'),dict(SPLIT=2,HAS_POST=K==4096,WORK=work))
 for split in (1,2,4):
  comp(m._dense_streamk,dict(A='*bf16',W='*bf16',PART='*fp32',T='i32'),dict(K=K,BM=64,BN=128,BK=64,SPLIT=split,GRID=264))
  comp(m._dense_streamk_boundary,dict(PART='*fp32',STREAM='*fp32',NEW='*fp32',OUT='*bf16',POST='*fp32',ALPHA='*fp32',BETA='*fp32',PRE='*fp32',T='i32'),dict(SPLIT=split,HAS_POST=K==4096,WORK=0))
comp(m._dense_reslot,dict(E='*i32',POS='*i32',COUNTS='*i32'),dict(N=3012*8,BLOCK=256))
(p.parent/'dev/dense_fusion_sm90.json').write_text(json.dumps(rows,indent=2))
print('SM90 dense fusion compile PASS',len(rows),flush=True)
