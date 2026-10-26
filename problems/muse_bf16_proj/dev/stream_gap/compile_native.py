"""Compile the actual generated generic kernels and the retained NF router for sm90a."""
import importlib.util,json,hashlib,sysconfig,subprocess,re
from pathlib import Path
import torch
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('sg_native',H/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
units=[]
class Mock:pass
def capture(name,cpp_sources='',cuda_sources='',**kw):units.append((name,cuda_sources));return Mock()
for bf in (False,True):
 ns=m._nf_vendor();src=ns['V18S_CUDA_TEMPLATE'];src=src.replace('using OutT = float;','using OutT = __nv_bfloat16;') if bf else src;ns['V18S_CUDA_TEMPLATE']=src;ns['load_inline']=capture
 insts=[]
 for t in (1,8,16,32):
  for name,n,k in [('qkv_gate',2304,8192),('o',8192,1024),('pooler',4096,8192),('unpooler',1024 if t<=8 else 8192,4096)]:insts.append((1,max(1,t//8),1,1,0,0,256,0,n,k,1,8,k//512,1,6))
 ns['v18s_build'](list(dict.fromkeys(insts)),'sg_compile_'+str(bf))
ns={'__name__':'sg_router_native'};exec(m.SG_ROUTER_SOURCE,ns);ns['load_inline']=capture;ns['rt2_build_router'](ns['rt2_ROUTER_CONFIGS'])
units.append(('sg_probe',(H/'probe.cu').read_text()))
out=H/'native';out.mkdir(exist_ok=True)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=sg_compile','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']]
flags += [arg for path in cpp_extension.include_paths(device_type='cuda') for arg in ('-isystem',path)]
rows=[]
for name,src in units:
 f=out/(name+'.cu');f.write_text(src);p=subprocess.run(['nvcc',*flags,'-c',str(f),'-o',str(f.with_suffix('.o'))],capture_output=True,text=True);log=p.stdout+p.stderr;f.with_suffix('.log').write_text(log)
 row=dict(name=name,source_sha256=hashlib.sha256(src.encode()).hexdigest(),passed=p.returncode==0,registers=list(map(int,re.findall(r'Used (\d+) registers',log))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),error=log[-2500:] if p.returncode else '')
 rows.append(row);print(row,flush=True)
(H/'native_validation.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest(),units=rows),indent=2)+'\n');assert all(r['passed'] for r in rows)
