import importlib.util,json,hashlib,sysconfig,subprocess,re
from pathlib import Path
import torch
from torch.utils import cpp_extension
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
H=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('sg_follow_compile',H/'followup_candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
out=H/'native';out.mkdir(exist_ok=True);source=m.sgo_source();p=out/'o1_fused.cu';p.write_text(source)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=sg_compile','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']];flags += [a for path in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',path)]
c=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=c.stdout+c.stderr;p.with_suffix('.log').write_text(log);assert c.returncode==0,log[-3000:]
rows=[dict(name='o1_fused',source_sha256=hashlib.sha256(source.encode()).hexdigest(),passed=True,registers=list(map(int,re.findall(r'Used (\d+) registers',log))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log))]
old=driver.active.get_current_target
try:
 driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
 for t in (128,256):
  for n,k in ((2304,8192),(8192,1024),(4608,8192),(8192,4096)):
   x=torch.empty((t,k),device='cuda',dtype=torch.bfloat16);w=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
   for bt,bw,bk,ns,nwarps,s in m.SG_REUSE_CONFIGS[t]:
    partial=torch.empty((s,t,n),device='cuda');xd=m._VDSK_TD.from_tensor(x,[bt,bk]);wd=m._VDSK_TD.from_tensor(w,[bw,bk]);name=f'reuse_{t}_{n}_{k}_{bw}_{s}'
    kernel=m._sg_reuse_gemm.warmup(x,w,partial,xd,wd,t,n,k,s,BT=bt,BW=bw,BK=bk,NS=ns,num_warps=nwarps,num_stages=ns,launch_pdl=True,grid=(t//bt*(n//bw),s))
    ptx=out/(name+'.ptx');ptx.write_text(kernel.asm['ptx']);assert 'wgmma.mma_async' in kernel.asm['ptx'] and 'cp.async.bulk.tensor' in kernel.asm['ptx']
    assembled=subprocess.run(['ptxas','-arch=sm_90a','-v',str(ptx),'-o',str(ptx.with_suffix('.cubin'))],capture_output=True,text=True);alog=assembled.stdout+assembled.stderr;ptx.with_suffix('.log').write_text(alog);assert assembled.returncode==0,alog
    row=dict(name=name,passed=True,shared=kernel.metadata.shared,registers=list(map(int,re.findall(r'Used (\d+) registers',alog))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',alog),ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest());rows.append(row);print(row,flush=True)
finally:driver.active.get_current_target=old
(H/'followup_native.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256((H/'followup_candidate.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n');assert all(all(set(v)=={'0'} for v in r['stack_spills']) for r in rows)
