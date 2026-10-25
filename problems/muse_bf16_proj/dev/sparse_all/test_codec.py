"""Independent timed-unpack proof: exact BF16 words including signed zero and escapes."""
import importlib.util,json,math,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('all_codec_candidate',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
g=torch.Generator(device='cuda').manual_seed(61791);rows=[]
shapes={'qkv_gate':(2304,8192),'o':(8192,1024),'gate':(1024,8192),'pooler':(4096,8192),'unpooler':(8192,4096),'router_hi':(256,8192),'router_lo':(256,8192)}
for name,(n,k) in shapes.items():
 w=(torch.randn((n,k),device='cuda',generator=g)/math.sqrt(k)).bfloat16()
 w[3,139]=0.;w[11,211]=-0.;w[19,27]=3.;w[23,39]=torch.finfo(torch.bfloat16).tiny
 c=m.cd_encode(w);b,stats=m.dc_encode_cd(c)
 out=torch.empty_like(w);m.dc_unpack(b,out);torch.cuda.synchronize()
 neq=int((out.view(torch.int16)!=w.view(torch.int16)).sum());assert neq==0,(name,neq)
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):m.dc_unpack(b,out)
 out.zero_();graph.replay();torch.cuda.synchronize();assert torch.equal(out.view(torch.int16),w.view(torch.int16))
 print('PASS',name,stats,flush=True);rows.append(dict(matrix=name,shape=(n,k),exact=True,stats=stats))
(H/'codec_validation.json').write_text(json.dumps(dict(passed=True,candidate_sha256=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
