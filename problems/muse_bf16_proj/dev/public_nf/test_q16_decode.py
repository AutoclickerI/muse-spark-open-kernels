"""Run the port's actual qkv16 decoder/MMA locally with CTA-local K reduction.
Only S4 DSM is replaced with S1/WK4/KU32; production packing, metadata, X mapping,
FP32 stores, template MT2, native host wrapper and PDL all execute unchanged.
"""
import runpy,time,json,hashlib
from pathlib import Path
import torch
H=Path(__file__).resolve().parent
p=runpy.run_path(str(H/'port_body.py'));ns=p['_nf_vendor']()
cu=ns['b2q_tt_v10_m32x__Q_CUDA']
for fmt in (0,1):
 a=f'<{fmt},2,1,1,5,0,128,0,0,8192,1,4,8,4,7,1,0>'
 assert a in cu
 cu=cu.replace(a,f'<{fmt},2,1,1,0,0,128,0,0,8192,1,4,32,1,7,1,0>')
ns['b2q_tt_v10_m32x__Q_CUDA']=cu
p['nf_prepare'].__globals__['_nf_vendor']=lambda:ns
p['NF_SMALL'].clear();p['NF_SMALL'][16,'qkv_gate']=('q16',(1,4,32,1),2,1,0,128,())
g=torch.Generator(device='cuda').manual_seed(14357)
shapes={'qkv_gate':(2304,8192),'pooler':(4096,8192),'router':(256,8192),'o':(8192,1024),'unpooler':(8192,4096)}
w={n:torch.randn(s,device='cuda',generator=g).to(torch.float32 if n=='router' else torch.bfloat16) for n,s in shapes.items()}
w['qkv_gate'][19,27]=3.;w['qkv_gate'][3,139]=0.
st=p['nf_prepare'](w,{(16,'qkv_gate')});records=[]
for seed in (61791,14357,92003):
 g.manual_seed(seed);x=torch.randn((16,8192),device='cuda',generator=g).bfloat16();x[-1]=0.;x[-1,27]=16.
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):y,_=p['nf_project'](x,'qkv_gate',st)
 graph.replay();torch.cuda.synchronize();ref=x.double()@w['qkv_gate'].double().T
 e=float((y[0].double()-ref).norm()/ref.norm());print('PASS',seed,e,flush=True);assert e<3e-5
 records.append(dict(seed=seed,error=e))
(H/'q16_decode_validation.json').write_text(json.dumps(dict(passed=True,adapter_sha256=hashlib.sha256((H/'port_body.py').read_bytes()).hexdigest(),standin='CTA-local reduction replaces S4 DSM only',records=records),indent=2)+'\n')
