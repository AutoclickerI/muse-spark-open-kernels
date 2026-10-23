"""Execute FA3 dispatch, tensor/tuple return handling and graph capture on sm_120."""
from pathlib import Path
import sys,ast,types
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root),str(root.parents[1]/'tools')]
import local_eval as E
ref=E.load_module('glue_reference',root/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**k:None
source=ast.parse((root/'cand_xqa_diag.py').read_text())
assert isinstance(source.body[-1],ast.If)
source.body.pop() # Explicitly exercise the Hopper dispatch below without running the tuner.
m=types.ModuleType('xqa_glue');exec(compile(source,str(root/'cand_xqa_diag.py'),'exec'),m.__dict__)
ctx=128
as_tuple=False

def standin(**kw):
 q=kw['q'];cfg=state['config'];batch=cfg['batch'];hq=cfg['num_q_heads'];hkv=cfg['num_kv_heads'];ps=cfg['page_size']
 ix=torch.arange(ctx+8,device=q.device)
 rows=kw['page_table'][:,ix//ps].long()*ps+ix%ps
 k=kw['k_cache'].view(-1,hkv,64)[rows].float().repeat_interleave(8,dim=2)
 v=kw['v_cache'].view(-1,hkv,64)[rows].float().repeat_interleave(8,dim=2)
 if 'k_descale' in kw:k*=kw['k_descale'][:,None,:,None].repeat_interleave(8,dim=2)
 if 'v_descale' in kw:v*=kw['v_descale'][:,None,:,None].repeat_interleave(8,dim=2)
 score=torch.einsum('blhd,bkhd->bhlk',q.view(batch,8,hq,64).float(),k)*kw['softmax_scale']
 pos=ctx+torch.arange(8,device=q.device)
 mask=ix[None,:]<=pos[:,None]
 if cfg['window']:mask &= ix[None,:]>=pos[:,None]-cfg['window']+1
 score=score.masked_fill(~mask[None,None],float('-inf'))
 out=torch.einsum('bhlk,bkhd->blhd',score.softmax(-1),v).reshape(batch*8,hq,64).to(torch.bfloat16)
 return (out,torch.empty(0,device=q.device)) if as_tuple else out
m.flash_attn_with_kvcache=standin
for layer,page,scale in [('full',1,1.),('sliding',1,1.),('full',64,.015625),('sliding',64,.015625)]:
 state=ref.generate_input(batch=2,ctx=ctx,heads='tp8',layer=layer,q_len=8,page=page,kv_scale=scale,seed=81)
 m._XPLAN[(2,bool(state['config']['window']))]='fa3'
 for as_tuple in (False,True):
  graph,out=E.capture(m.kernel_fn,state);graph.replay();torch.cuda.synchronize()
  ok,msg=ref.check_implementation(state,ref.collect(state,out))
  print(layer,page,scale,'tuple',as_tuple,ok,msg,flush=True)
  assert ok,msg
# Exercise all tuner control-flow, graph, selection, and FA3 branches using small
# synthetic inputs; the exact 80K synthetic allocation is independently screened.
original=m._synthetic
m._synthetic=lambda b,w:ref.generate_input(batch=1 if b==1 else 2,ctx=128,heads='tp8',layer='sliding' if w else 'full',q_len=8,page=1,seed=84)
m._fa3_verify=lambda data:m._xqa_verify(data) # Portable callable stand-in for tuner API only.
m._diagnose(True)
assert len(m._XPLAN)>=6
print('PASS: Hopper FA3 glue, tensor/tuple, scaling, page1/page64, sliding/full and tuner branches')
