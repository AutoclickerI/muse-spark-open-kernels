"""Save actual captured graph edge types, kernel node names and an overlap timeline."""
from pathlib import Path
import importlib.util,sys,json,collections
import torch
from cuda.bindings import runtime as rt
p=Path(__file__).resolve().parents[1];sys.path[:0]=[str(p)]
import reference as r
if torch.cuda.get_device_capability()!=(9,0):r._fa3_scheduler_metadata=lambda *a,**k:None
sp=importlib.util.spec_from_file_location('c',p/(sys.argv[1] if len(sys.argv)>1 else 'cand_chain6.py'))
c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
s=c.prepare(r.generate_weights(r.WEIGHTS_SEED))
for batch,ql in ((1,1),(1,8),(16,1),(32,1),(16,8),(32,8)):
 d=r.generate_input(batch=batch,q_len=ql,ctx=80000,layer='sliding',dist='soft',page=1,seed=77)
 st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(st):
  for _ in range(3):c.kernel_fn(d,s)
 torch.cuda.current_stream().wait_stream(st)
 g=torch.cuda.CUDAGraph();g.enable_debug_mode()
 with torch.cuda.graph(g):o=c.kernel_fn(d,s)
 g.debug_dump(str(p/f'dev/graph_{batch}_{ql}.dot'))
 status,src,dst,edges,n=rt.cudaGraphGetEdges(g.raw_cuda_graph(),100)
 if status!=rt.cudaError_t.cudaSuccess:raise RuntimeError(status)
 erows=[{'from':int(a),'to':int(b),'type':int(e.type),'from_port':int(e.from_port),'to_port':int(e.to_port)} for a,b,e in zip(src[:n],dst[:n],edges[:n])]
 (p/f'dev/graph_{batch}_{ql}_edges.json').write_text(json.dumps(erows,indent=2))
 print('edge types',collections.Counter(e['type'] for e in erows),flush=True)
 torch.cuda.synchronize()
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
  g.replay();torch.cuda.synchronize()
 ev=[x for x in prof.events() if x.device_type==torch.autograd.DeviceType.CUDA]
 lo=min(x.time_range.start for x in ev)
 rows=[{'name':x.name,'start_us':x.time_range.start-lo,'duration_us':x.device_time} for x in ev]
 (p/f'dev/graph_{batch}_{ql}.json').write_text(json.dumps(rows,indent=2))
 print(batch,ql,'GPU activities',len(ev),collections.Counter(x.name.split('<')[0] for x in ev),flush=True)
