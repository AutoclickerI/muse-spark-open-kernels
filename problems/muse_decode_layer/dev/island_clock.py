from pathlib import Path
import importlib.util,sys,torch,statistics
p=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(p)]
sp=importlib.util.spec_from_file_location('candidate',p/'cand_persistent.py')
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
import reference as r
r._fa3_scheduler_metadata=lambda *a,**kw:None
w=r.generate_weights(r.WEIGHTS_SEED);s=c.prepare(w)
flush=torch.empty(1<<30,dtype=torch.uint8,device='cuda')
for t in (1,8,16,32):
 d=r.generate_input(batch=t,ctx=3000,layer='sliding',seed=77)
 inp=torch.randn(t,1024,device='cuda',dtype=torch.bfloat16)
 for prefix in (True,False):
  if not prefix: inp=torch.randn(t,4096,device='cuda',dtype=torch.bfloat16)
  cnt=torch.zeros(2+5*t,device='cuda',dtype=torch.int32)
  stamps=torch.zeros(5,device='cuda',dtype=torch.int64)
  for _ in range(2): cnt.zero_(); c._island(inp,d['stream'],s,d,cnt,prefix,stamps)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(stream):
   cnt.zero_();c._island(inp,d['stream'],s,d,cnt,prefix,stamps)
  torch.cuda.current_stream().wait_stream(stream)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):
   cnt.zero_();out=c._island(inp,d['stream'],s,d,cnt,prefix,stamps)
  ts=[]
  for i in range(20):
   flush.sum();g.replay();torch.cuda.synchronize();v=stamps.tolist()
   ts.append([(v[j]-v[i])*.001 for i,j in ([(0,1),(1,2),(2,3),(3,4)] if prefix else [(0,1),(1,4)])])
  print(t,'prefix' if prefix else 'suffix',[round(statistics.median(x[i] for x in ts),2) for i in range(4 if prefix else 2)],flush=True)
