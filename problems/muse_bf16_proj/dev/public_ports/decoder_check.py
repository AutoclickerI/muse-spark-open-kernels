"""Execute the actual integer decoder on sm120, including every BF16 bit pattern."""
import importlib.util,json,hashlib,sys
from pathlib import Path
import torch,triton,triton.language as tl
H=Path(__file__).resolve().parent;s=importlib.util.spec_from_file_location('public_decode_actual',H/'out/source.py');m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
@triton.jit
def probe(W,B,O,E,Y,N:tl.constexpr,K:tl.constexpr,DECODE:tl.constexpr):
 r=tl.program_id(0)*16+tl.arange(0,16);k=tl.program_id(1)*128+tl.arange(0,128);v=DECODE(W,B,O,E,r,tl.program_id(1)*128,K,128);tl.store(Y+r[:,None]*K+k[None,:],v)
rows=[]
for label,w in [('all_bits',torch.arange(65536,device='cuda',dtype=torch.int32).to(torch.int16).view(torch.bfloat16).view(256,256)),('gaussian',torch.randn(256,8192,device='cuda',dtype=torch.bfloat16))]:
 p=m.pp_encode(w)
 for name,q,ref in ((label,p,w),(label+'_slice',m.pp_slice(p,16,240),w[16:240])):
  y=torch.empty_like(ref);probe[(ref.shape[0]//16,ref.shape[1]//128)](q['words'],q['base'],q['offset'],q['values'],y,*ref.shape,DECODE=m._pp_c12,num_warps=4);torch.cuda.synchronize();assert torch.equal(y.view(torch.int16),ref.view(torch.int16)),name;rows.append(dict(case=name,weights=ref.numel(),bytes=sum(q[k].numel()*q[k].element_size() for k in ('words','base','offset','values'))))
(H/'decoder_validation.json').write_text(json.dumps(dict(passed=True,cases=rows,actual_device_decoder=True,encode_source_sha256=hashlib.sha256((H/'coded.py.inc').read_bytes()).hexdigest()),indent=2)+'\n');print('PASS actual decoder',rows)
