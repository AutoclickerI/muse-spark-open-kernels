"""Minimal reproducer for captured cuBLAS workspaces across output-dtype paths."""
import torch
from pathlib import Path
import importlib.util
p=Path(__file__).resolve().parents[1]/'cand_portfolio.py'
s=importlib.util.spec_from_file_location('c',p);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
f=torch.zeros(128*1024*1024,device='cuda',dtype=torch.uint8)
for layer,n,k in [('router',256,8192),('qkv',2304,8192)]:
 w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)
 lo=(torch.randn_like(w)*0.002).contiguous(); wt=w.T.contiguous(); lt=lo.T.contiguous()
 pair=torch.cat((w,lo),1).contiguous()
 for m in (128,256):
  x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
  def base():
   if layer=='router':
    y=torch.mm(x,wt,out_dtype=torch.float32); y.add_(torch.mm(x,lt,out_dtype=torch.float32)); return y
   return torch.nn.functional.linear(x,w)
  def variant():
   y=torch.mm(x,pair[:,:k].T if layer=='router' else w.T,out_dtype=torch.float32)
   if layer=='router': y.add_(torch.mm(x,pair[:,k:].T,out_dtype=torch.float32))
   out=torch.empty((m,n),device='cuda',dtype=torch.float32 if layer=='router' else torch.bfloat16)
   out.copy_(y); return out
  bg,bo=c._graph(base)
  for i in range(70):
   g,o=c._graph(variant); c._cold_times([g],f,3)
  print(layer,m,'trials passed',flush=True)
  gs=[];outs=[]
  for i in range(3):
   g,o=c._graph(variant);gs.append(g);outs.append(o)
  print(layer,m,c._cold_times([bg]+gs,f,9),flush=True)
