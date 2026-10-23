import importlib.util, sys, torch, statistics
spec=importlib.util.spec_from_file_location('candidate',sys.argv[1]);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
for t in (1,8,16,32,128):
 p=torch.randn(4,t,8192,device='cuda');st=torch.randn(t,8192,device='cuda')
 a=torch.rand(8192,device='cuda');b=torch.rand_like(a);w=torch.rand_like(a)
 out=torch.empty_like(st);sout=torch.empty_like(st);sync=torch.zeros(5*t,device='cuda')
 results=[]
 for chunks in (1,2,4,8,16):
  if chunks*t>128 and chunks>1:continue
  for warps in (4,8,16) if chunks==1 else (4,):
   def f():
    sync.zero_()
    c._launch(c._boundary_kernel,(t,chunks),p,4,p.stride(0),p.stride(1),st,sout,out,a,a,b,w,sync,sync[4*t:].view(torch.int32),1e-8,1e-5,NB=8192,H=8192,C=chunks,HAS_POSTW=False,num_warps=warps)
   for _ in range(2):f()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):f()
   ev1=torch.cuda.Event(enable_timing=True);ev2=torch.cuda.Event(enable_timing=True);ts=[]
   for _ in range(10):
    flush.zero_();ev1.record();g.replay();ev2.record();ev2.synchronize();ts.append(ev1.elapsed_time(ev2)*1e3)
   results.append((round(statistics.median(ts),2),chunks,warps))
 print(t,sorted(results),flush=True)
