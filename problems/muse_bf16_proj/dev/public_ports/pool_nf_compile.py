"""Run under tools/img: compile exact queued NF kernels and inspect PTX ordering."""
import hashlib,importlib.util,json,re,subprocess
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;O=H/'out'
spec=importlib.util.spec_from_file_location('p10_pool_source_capture',O/'whole_source.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);original=m._nf_vendor
class SourceCaptured(Exception):pass
def vendor():
 ns=original()
 def build(insts,label):
  (O/'pool_variants.cu').write_text(ns['v18s_cuda_source'](insts));raise SourceCaptured
 ns['v18s_build']=build;return ns
m._nf_vendor=vendor
try:m.pp_pool_prepare(vars(m),dict(pool_router=torch.empty((4608,8192),device='cuda',dtype=torch.bfloat16)))
except SourceCaptured:pass  # Deliberate source-only interception; no module has launched.
with (O/'nf_nvcc.log').open('w') as f:subprocess.run(['nvcc','-arch=sm_90a','-cubin','-lineinfo','-Xptxas=-v',str(O/'pool_variants.cu'),'-o',str(O/'pool_variants.cubin')],stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run(['nvcc','-arch=sm_90a','-ptx','-lineinfo',str(O/'pool_variants.cu'),'-o',str(O/'pool_variants.ptx')],check=True)
s=(O/'pool_variants.ptx').read_text();rows=[];log=(O/'nf_nvcc.log').read_text()
for part in s.split('.visible .entry ')[1:]:
 name=part.split('(')[0].strip()
 if not name.startswith('_Z9v18s_gemv'):continue
 body=part.split('// .globl',1)[0];params=list(map(int,re.findall(r'Li(\d+)E',name)));wait=body.find('griddepcontrol.wait');ldg=body.find('ld.global');assert wait>=0 and ldg>=0
 prop=log.split('Function properties for '+name+'\n',1)[1].split('ptxas info    : Compile time',1)[0]
 rows.append(dict(tokens=params[1]*8,stages=params[3],warps=params[11],OPT=params[-1],early=params[-1]&64!=0,PTX_global_load_before_wait=ldg<wait,zero_spill='0 bytes spill stores, 0 bytes spill loads' in prop,registers=int(re.search(r'Used (\d+) registers',prop).group(1)),function=name))
assert len(rows)==12 and all(x['early']==x['PTX_global_load_before_wait'] and x['zero_spill'] for x in rows)
out=dict(passed=True,source_sha256=hashlib.sha256((O/'pool_variants.cu').read_bytes()).hexdigest(),cubin_sha256=hashlib.sha256((O/'pool_variants.cubin').read_bytes()).hexdigest(),units=rows,limits='NVCC sm90a compile plus PTX issue-order proof: first global load precedes dependency wait only in OPT64 variants. All12 kernels zero spills. Offline SASS has no visible GDC mnemonic; no runtime issue/wait effect inferred. Native H200 effect remains queued/unmeasured.')
(H/'pool_nf_native_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('PASS12 NF sm90a kernels, zero spills, PTX issue ordering')
