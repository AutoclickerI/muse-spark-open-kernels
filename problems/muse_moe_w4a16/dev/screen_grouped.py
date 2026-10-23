"""Correctness/capture controls for every tile width and true large-M canonical-bank use."""
import importlib.util
import sys
from pathlib import Path
import torch
p=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(p))
import reference as R
from nvfp4 import random_weights
filename=next((arg for arg in sys.argv[1:] if arg.endswith('.py')), 'cand_rs.py')
spec=importlib.util.spec_from_file_location('cand',p/filename)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
gen=torch.Generator(device='cuda').manual_seed(189793)
w={}
for name,rows in (('w13',8192),('w2',4096)):
 q,s,g=random_weights(rows,4096,generator=gen,device='cuda',leading=(32,),wide_scales=True)
 w[name],w[name+'_sf'],w[name+'_gs']=q,s,g.expand(32).contiguous()
w['post_norm_w']=(torch.randn(4096,generator=gen,device='cuda')*.1).bfloat16()
st=c.prepare(w)
assert not any(key in st for key in ('blob13','sfb13','bank','humming','marlin'))
bank_bytes=st['w13p'].numel()+st['w2p'].numel()
print('Resident canonical packed bank:',bank_bytes,'bytes',flush=True)
torch.backends.cuda.matmul.allow_tf32=False

def check(data,out,samples=None):
 if samples is None:
  if not bool(((data['topk_ids']>=0)&(data['topk_ids']<32)).any()):
   assert bool((out==0).all());return 'exact zeros'
  ok,msg=R.check_implementation(data,out,w)
 else:
  ix=torch.tensor(samples,device='cuda')
  sub={**data,'hidden_states':data['hidden_states'][ix], 'topk_ids':data['topk_ids'][ix], 'topk_weights':data['topk_weights'][ix]}
  ok,msg=R.check_implementation(sub,out[ix],w)
 assert ok,msg
 return msg

if '--tiles' in sys.argv:
 data=R.generate_input(65,'ep8',828275)
 for n in (8,16,32,64,128,0):
  for stages in (2,3):
   out=c._grouped_call(data,st,n,stages)
   torch.cuda.synchronize()
   print('tile',n,'stages',stages,check(data,out),flush=True)
 # Capture once, change activations AND expert populations on every replay.
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=c._grouped_call(data,st)
 for distribution in ('hot','empty','normal','hot'):
  data['hidden_states'].normal_(generator=gen)
  if distribution=='hot':data['topk_ids'].copy_(torch.arange(8,device='cuda',dtype=torch.int32).expand(65,8))
  elif distribution=='empty':data['topk_ids'].fill_(60)
  else:data['topk_ids'].copy_(R.generate_input(65,'ep8',12442)['topk_ids'])
  graph.replay();torch.cuda.synchronize()
  print('routing replay',distribution,check(data,out),flush=True)
 del graph,out,data
 torch.cuda.empty_cache()

if '--large' in sys.argv:
 for m in (753,3012,16384):
  x=torch.randn((m,4096),generator=gen,device='cuda',dtype=torch.bfloat16)
  ids=torch.full((m,8),60,device='cuda',dtype=torch.int32);ids[:,0]=0
  data={'hidden_states':x,'topk_ids':ids,'topk_weights':torch.full((m,8),.125,device='cuda'),
        'config':{'local_count':32,'local_start':0,'d_hidden':4096,'d_expert':4096}}
  # One local expert on every token: exercises N128, int32 routes and 16K-token offsets.
  c._grouped_call(data,st)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):out=c._grouped_call(data,st)
  for replay in range(2):
   x.normal_(generator=gen);graph.replay();torch.cuda.synchronize()
   print('large',m,'replay',replay,check(data,out,[0,m//2,m-1]),flush=True)
  del graph,out,data,x,ids
  torch.cuda.empty_cache()
