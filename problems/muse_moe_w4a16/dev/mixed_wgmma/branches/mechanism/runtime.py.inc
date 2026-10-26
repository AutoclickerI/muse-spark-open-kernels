MECH_NAMES=('BASE','PACK','PREPACK','PACK3','EARLY3','MC2','PREMC2','DUPSYNC','DUPCTA')
DIAG_NS={}
def probe_module():
 if not hasattr(probe_module,'mod'):
  arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
  mod=load_inline(name='p3b_probe_'+hashlib.sha256(PROBE_CUDA.encode()).hexdigest()[:12],cpp_sources=PROBE_CPP,cuda_sources=PROBE_CUDA,functions=['probe_run','probe_info'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
  for k in range(4):
   for f in (0,1):mod.probe_info(k,f)
  probe_module.mod=mod
 return probe_module.mod

def mechanism_build():
 import ast,linecache
 if DIAG_NS:return
 act_build();probe_module()
 ns=dict(ACT_NS['ARMS']);ns['probe_module']=probe_module
 src=ast.get_source_segment(ACT_CELL,next(n for n in ast.parse(ACT_CELL).body if isinstance(n,ast.FunctionDef) and n.name=='activation_plan'))
 src=src.replace('def activation_plan(','def diagnostic_plan(').replace("arm in ('PACK','TMA1','MC2','MC4')",'True').replace('arm in ("PACK","TMA1","MC2","MC4")','True')
 src=src.replace('kind={"PACK":0,"TMA1":1,"MC2":2,"MC4":4,"W4N32":5,"W4N64":6}[arm]', 'pr=probe_module();kind={"PACK3":0,"EARLY3":1,"DUPSYNC":2,"DUPCTA":3}[arm]')
 src=src.replace('mod.ax_run(','pr.probe_run(')
 filename='<diagnostic-plan>';linecache.cache[filename]=(len(src),None,src.splitlines(True),filename);exec(compile(src,filename,'exec'),ns);DIAG_NS['plan']=ns['diagnostic_plan']

class MechanismGraph:
 def __init__(self,data,state,arm):
  self.data={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()};self.arm=arm
  args=(self.data,state['base'],state['native'],state['streams'])
  if arm=='BASE':p=ACT_NS['S18']['p3b_plan'](*args)
  elif arm in ('PACK','PREPACK','MC2','PREMC2'):
   p=ACT_NS['ARMS']['activation_plan'](*args,'MC2' if arm in ('MC2','PREMC2') else 'PACK')
  else:p=DIAG_NS['plan'](*args,arm)
  self.plan=p;self.pre=arm in ('PREPACK','PREMC2')
  for f in p['steps']:self.out=f()
  torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
  # Route excluded uniformly: atomic slot order must match pre-gathered rows.
  self.steps=p['steps'][1:] if arm=='BASE' or self.pre else (p['gather_only'],)+p['steps'][1:]
  with torch.cuda.graph(self.graph):
   for f in self.steps:self.out=f()
 def refresh(self,d,canonical=None):
  for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(d[k])
  if canonical is None:self.plan['steps'][0]()
  else:
   dst,src=self.plan['owners'],canonical.plan['owners']
   for i in (4,5,6,8):dst[0][i].copy_(src[0][i])
   for dt,st in zip(dst[6],src[6]):dt.copy_(st)
   dst[5].copy_(src[5])
  if self.pre:self.plan['gather_only']()
  torch.cuda.synchronize()
 def replay(self):self.graph.replay();return self.out
