import importlib.util,json,pathlib,sys,torch
H=pathlib.Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'problems/muse_decode_layer'))
spec=importlib.util.spec_from_file_location('ewdense',H/'U_trial.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);mod=m.ew_dense_module();rows=[]
for t,n,k in ((1,1024,4096),(8,8192,1024),(16,8192,4096),(32,4608,8192),(128,8192,4096),(256,2304,8192)):
 x=torch.randn((t,k),device='cuda',dtype=torch.bfloat16);w=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);want=x.float()@w.float().T
 for mode in (0,1):
  y=torch.empty_like(want);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):mod.ew_dense_run(x.data_ptr(),w.data_ptr(),y.data_ptr(),t,n,k,mode,torch.cuda.current_stream().cuda_stream)
  g.replay();torch.cuda.synchronize();err=float((y-want).norm()/want.norm());assert err<.001,(t,n,k,mode,err)
  rows.append(dict(t=t,n=n,k=k,mode=mode,error=err,resources=mod.ew_info(64 if t>32 else 16,mode)))
(H/'dense_validation.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',len(rows),'dense cases',flush=True)
