"""Cross-compile every emitted fusion specialization for H200, without launching."""
from pathlib import Path
import importlib.util,json,time
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
spec=importlib.util.spec_from_file_location('glue90',Path(__file__).with_name('projection_glue.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
seen=set();rows=[];start=time.monotonic()
for m in g.GB_TOKENS:
 for phase in ('b1','pool','b2'):
  for bt,bn,splits,bk,warps in g.gb_choices(m,phase):
   tb=max(16,triton.next_power_of_2(bt))
   if phase=='pool':
    fn=g._gb_pool_route
    sig={n:'*fp32' for n in ('PRE','BIAS','P','RP','WTS')}
    sig.update({n:'*i32' for n in ('LSTART','CNT','IDS','LIDS')})
    sig.update(X='*bf16',W='*bf16',Y='*bf16',M='i32')
    const=dict(BN=bn,BK=bk,KS=8192//splits,BT=bt,TB=tb,WARPS=warps)
   else:
    fn=g._gb_boundary_gemv;n=1024 if phase=='b2' and m<=8 else 8192;k=1024 if phase=='b1' else 4096
    sig={name:'*fp32' for name in ('STREAM','ALPHA','BETA','POST','PRE','P','SOUT')}
    sig.update(X='*bf16',W='*bf16',OUT='*bf16',CNT='*i32',M='i32')
    const=dict(N=n,K=k,BN=bn,BK=bk,KS=k//splits,BT=bt,TB=tb,POSTW=phase=='b2',WARPS=warps)
   key=(fn.__name__,tuple(const.items()))
   if key in seen:continue
   seen.add(key);sig.update({n:'constexpr' for n in const})
   compiled=triton.compile(ASTSource(fn,signature=sig,constexprs=const),
       target=GPUTarget('cuda',90,32),options=dict(num_warps=warps,num_stages=2,enable_fp_fusion=False))
   assert compiled.metadata.shared<=227328,compiled.metadata
   row={'kernel':fn.__name__,'constants':const,'smem':compiled.metadata.shared,
        'wgmma':compiled.asm['ptx'].count('wgmma.mma_async'),'cubin_bytes':len(compiled.asm['cubin'])}
   rows.append(row);print(row,flush=True)
print('SM90_COMPILE_PASS',len(rows),'elapsed_s',time.monotonic()-start,flush=True)
Path('runs/screen/verify_proj/glue_sm90.json').write_text(json.dumps(rows,indent=2)+'\n')
