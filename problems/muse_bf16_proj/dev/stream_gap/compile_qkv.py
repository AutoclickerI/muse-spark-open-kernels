"""Compile the exact Hopper kernel and exercise its host cluster dispatch locally."""
import importlib.util,hashlib,json,subprocess,sysconfig,re
from pathlib import Path
import torch
from torch.utils import cpp_extension
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('sgq_compile',H/'qkv_candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ns,insts,plans=m.sgq_source();source=ns['v18s_cuda_source'](insts)
folder=H/'native';folder.mkdir(exist_ok=True);p=folder/'qkv_dispatch.cu';p.write_text(source)
flags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','-DTORCH_EXTENSION_NAME=sgq_compile','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v','-isystem',sysconfig.get_paths()['include']]
flags += [a for path in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',path)]
run=subprocess.run(['nvcc',*flags,'-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=run.stdout+run.stderr;p.with_suffix('.log').write_text(log);assert run.returncode==0,log[-3000:]
# Leave the native host wrapper, configuration, signature and selected instance
# intact. A no-op CUDA kernel is the only substitute for Hopper DSM.
begin=source.index('{',source.index('v18s_gemv('))
end=source.index('\nstatic int v18s_smem_optin()',begin)
stub=source[:begin]+'{ }\n'+source[end:]
# sm120 lacks cluster launch. Still validate all original pre-launch conditions,
# then use the same grid/block and argument list without the cluster attribute.
stub=stub.replace('cudaLaunchKernelEx(&lc, kern,','lc.attrs=nullptr;lc.numAttrs=0;\n  cudaLaunchKernelEx(&lc, kern,')
mod=ns['load_inline'](name='sgq_glue_'+hashlib.sha256(stub.encode()).hexdigest()[:12],cpp_sources=ns['V18S_CPP'],cuda_sources=stub,functions=['v18s_launch','v18s_router_launch'],extra_cuda_cflags=['-O3',ns['_nf_arch_flag']()],no_implicit_headers=True,with_pytorch_error_handling=False)
# Launch-bound fields and all template/shape guards execute. The task arithmetic
# is independently screened through S1 on every same decoder specialization.
device='cuda';x=torch.empty((16,8192),device=device,dtype=torch.bfloat16);out=torch.empty((16,2304),device=device);ptr=torch.empty(16,device=device,dtype=torch.int32)
count=0
for (t,n,arm),(idx,cfg) in plans.items():
    rc=mod.v18s_launch(idx,ptr.data_ptr(),x.data_ptr(),out.data_ptr(),ptr.data_ptr(),t,n,8192,*cfg,0,0,torch.cuda.current_stream().cuda_stream,0)
    assert rc==0,(t,n,arm,rc);count+=1
torch.cuda.synchronize()
row=dict(passed=True,candidate_sha256=hashlib.sha256((H/'qkv_candidate.py').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(source.encode()).hexdigest(),units=len(insts),host_dispatches=count,registers=list(map(int,re.findall(r'Used (\d+) registers',log))),stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),standin='no-op kernel and cluster attribute removed locally; native host guards/grid/ABI retained')
(H/'qkv_native.json').write_text(json.dumps(row,indent=2)+'\n');print(row)
