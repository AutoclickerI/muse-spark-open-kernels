"""Compile all selectable glue geometries for Hopper, including spilled registers."""
from pathlib import Path
import importlib.util,json
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
p=ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py';sp=importlib.util.spec_from_file_location('glue_resources',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=[]
for nb,s in ((8192,1),(8192,2),(8192,8),(1024,1),(1024,8)):
 for post in (False,True):
  for nw in (8,16):
   for replicas in (1,4,16):
    sig={'BR':'*bf16' if s==1 else '*fp32','STREAM':'*fp32','SOUT':'*fp32','OUT':'*bf16','POSTW':'*fp32','ALPHA':'*fp32','BETA':'*fp32','PREW':'*fp32'}
    const=dict(S=s,BS=nb*8,RS=nb,NB=nb,HAS_POSTW=post,REPLICAS=replicas,USE_PDL=True)
    k=triton.compile(ASTSource(m._boundary_fast,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options={'num_warps':nw})
    assert k.metadata.shared<232448
    ptx=k.asm['ptx'];start=ptx.index('griddepcontrol.launch_dependents');wait=ptx.index('griddepcontrol.wait')
    assert start<wait and 'ld.global' in ptx[start:wait]
    rows.append(dict(kernel='boundary',nb=nb,s=s,post=post,nw=nw,replicas=replicas,shared=k.metadata.shared))
for s in (1,4,8,16):
 for nw in (4,8):
  sig={p:'*fp32' for p in ('PR','BIAS','PREW','WTS')};sig.update({p:'*i32' for p in ('LSTART','IDS','LIDS')});sig['X']='*bf16'
  const=dict(S=s,PS=8*4608,RS=4608,REPLICAS=1,USE_PDL=True)
  k=triton.compile(ASTSource(m._route_fast,signature=sig,constexprs=const),target=GPUTarget('cuda',90,32),options={'num_warps':nw})
  assert k.metadata.shared<232448
  ptx=k.asm['ptx'];start=ptx.index('griddepcontrol.launch_dependents');wait=ptx.index('griddepcontrol.wait')
  assert start<wait and 'ld.global' in ptx[start:wait]
  rows.append(dict(kernel='route',s=s,nw=nw,shared=k.metadata.shared))
(HERE/'hopper_resources.json').write_text(json.dumps(rows,indent=2)+'\n');print('HOPPER_GLUE_COMPILES',len(rows),'max_shared',max(r['shared'] for r in rows),flush=True)
