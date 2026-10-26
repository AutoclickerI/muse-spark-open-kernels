"""Compile the actual changed Triton glue and existing gate for sm90, without launching it."""
import importlib.util,json,subprocess,sys
from pathlib import Path
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'problems/muse_decode_layer'))
def module(path,name):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
base=module(H/'out/base.py','ew_tb');new=module(H/'ALL_trial.py','ew_tn')
out=H/'native';out.mkdir(exist_ok=True);rows=[]
for label,m in (('BASE',base),('ALL',new)):
 for fname in ('_boundary_fast','_boundary_late'):
  for post in (False,True):
   for nw in (4,8,16):
    fn=getattr(m,fname);constants=dict(S=1,BS=8192*128,RS=8192,NB=8192,HAS_POSTW=post,USE_PDL=True)
    if fname=='_boundary_fast':constants['REPLICAS']=1
    sig={a:('*bf16' if a=='OUT' else '*fp32') for a in fn.arg_names if a not in constants}
    k=triton.compile(ASTSource(fn,sig,constants),target=GPUTarget('cuda',90,32),options=dict(num_warps=nw))
    name=f'{label}{fname}_{int(post)}_{nw}';p=out/(name+'.cubin');p.write_bytes(k.asm['cubin'])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p)],text=True);p.with_suffix('.sass').write_text(sass)
    res=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p)],text=True)
    rows.append(dict(name=name,metadata=k.metadata._asdict(),resources=res))
fn=base._gate_split_kernel;constants=dict(N=1024,BN=base.GATE_SPLIT_CFG[0],BK=128,TB=16,NS=base.GATE_SPLIT_CFG[2]);sig={a:('*fp32' if a=='P' else '*i32' if a=='CNT' else '*bf16' if a in ('X','W','Y','G') else 'i32') for a in fn.arg_names if a not in constants}
k=triton.compile(ASTSource(fn,sig,constants),target=GPUTarget('cuda',90,32),options=dict(num_warps=4));p=out/'GATE.cubin';p.write_bytes(k.asm['cubin']);p.with_suffix('.sass').write_text(subprocess.check_output(['cuobjdump','--dump-sass',str(p)],text=True))
(H/'triton_validation.json').write_text(json.dumps(rows,indent=2,default=str)+'\n');print('PASS',len(rows)+1,'sm90 Triton builds')
