"""Each board's real task and graph checks; local-only Hopper launch stand-ins."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
import static_cache
static_cache.install(local_eval)
role=sys.argv[1];forced='--forced' in sys.argv
loader=local_eval.load_module;calls={}
def load(name,path):
 m=loader(name,path)
 if name!='submission':return m
 if role=='attention':
  sp=importlib.util.spec_from_file_location('p12_local_attn',H.parent/'s15/local_standins.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);f.install(m)
  if forced:
   sp=importlib.util.spec_from_file_location('p12_verify_best',H.parent/'verify_best.py');f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);f.install(m,False)
 if role=='projection' and forced:
  m._P12_FORCE=True
  class VD:
   def __getitem__(self,grid):
    def run(x,w,p,xd,wd,mm,n,k,s,**kw):
     assert p.shape==(s,mm,n);p.zero_();p[0].copy_(m.torch.mm(x,w.T,out_dtype=m.torch.float32))
     calls['vdsk']=calls.get('vdsk',0)+1
    return run
  m._vdsk_gemm=VD()
 for label in {'moe':('smallm_experts','nf_plan','v9_kernel_fn'),'attention':('_cluster_attention','_layer_partial_port','_vq8_attention','_attn_port_variant'),'projection':('nf_project','_project_variant')}[role]:
  fn=getattr(m,label)
  def counted(*a,_fn=fn,_name=label,**kw):
   row=calls.setdefault(_name,dict(eager=0,captured=0));row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
   return _fn(*a,**kw)
  setattr(m,label,counted)
 return m
local_eval.load_module=load
out=H/'screen'/role/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/('cand_s15_'+role+'.py')),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
try:local_eval.main()
finally:(out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
