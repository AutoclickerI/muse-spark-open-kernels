"""Independent P*ones/rescale probe, plus O3 Hopper compilation and SASS gates."""
import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sysconfig
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline,include_paths
H=Path(__file__).resolve().parent;out=H/'pvsum';native=out/'native';native.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('pvsum_candidate',out/'candidate.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
probe=(H/'pvsum_probe.cu.inc').read_text();src=m.VQ8_CUDA+probe
tag=hashlib.sha256(src.encode()).hexdigest()[:12];arch=''.join(map(str,torch.cuda.get_device_capability()))
mod=load_inline(name='pvsum_probe_'+tag+'_'+arch,cpp_sources='#include <cstdint>\nvoid pv_den_probe(int64_t,int64_t,int64_t,int64_t);',
    cuda_sources=src,functions=['pv_den_probe'],extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
rows=[]
for seed in (68137,77051,94973):
    torch.manual_seed(seed);p=(torch.rand((2,64,128),device='cuda')*448).to(torch.float8_e4m3fn)
    alpha=torch.where(torch.arange(64,device='cuda')%3==0,.5,1.).float();y=torch.empty((128,4),device='cuda')
    mod.pv_den_probe(p.data_ptr(),alpha.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
    def run():mod.pv_den_probe(p.data_ptr(),alpha.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):run()
    g.replay();g.replay();torch.cuda.synchronize()
    lane=torch.arange(128,device='cuda')%32;w=torch.arange(128,device='cuda')//32
    row=16*w+lane//4;ref=p[0].float().double().sum(-1)*alpha.double()+p[1].float().double().sum(-1)
    want=torch.stack((ref[row],ref[row],ref[row+8],ref[row+8]),-1)
    err=float(((y.double()-want).abs()/want.abs().clamp_min(1e-12)).max());assert err<1e-6,err
    rows.append(dict(seed=seed,max_relative=err,graph_replays=2))
cu=native/'pvsum.cu';cu.write_text(src);obj=cu.with_suffix('.o')
args=['nvcc','-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr','-Xptxas=-v',
      '-DTORCH_EXTENSION_NAME=pvsum_native','-DTORCH_API_INCLUDE_EXTENSION_H','-isystem',sysconfig.get_paths()['include']]
args += [a for p in include_paths(device_type='cuda') for a in ('-isystem',p)]
p=subprocess.run([*args,'-c',str(cu),'-o',str(obj)],capture_output=True,text=True)
log=p.stdout+p.stderr;(native/'compile.log').write_text(log);assert p.returncode==0,log[-4000:]
# S15's unchanged combine already spills at its launch_bounds(256,4). Compare
# exact baseline resources; reject any new spill in the changed main kernels.
old=(H.parents[3]/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text()
old_cuda=next(n.value.value for n in ast.parse(old).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA')
bc=native/'baseline.cu';bc.write_text(old_cuda)
b=subprocess.run([*args,'-c',str(bc),'-o',str(bc.with_suffix('.o'))],capture_output=True,text=True)
blog=b.stdout+b.stderr;(native/'baseline.log').write_text(blog);assert b.returncode==0,blog[-4000:]
def resources(text):
    pattern=r"Compiling entry function '([^']+)'.*?\n.*?\n\s+(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads.*?Used (\d+) registers"
    return [dict(symbol=n,stack=int(st),stores=int(ss),loads=int(sl),regs=int(r)) for n,st,ss,sl,r in re.findall(pattern,text,re.S)]
rr,br=resources(log),resources(blog)
main=[x for x in rr if 'vq8_main' in x['symbol']];assert len(main)==3
no_spills=all(x['stores']==x['loads']==0 for x in main)
# This is a performance diagnostic, not a math/correctness criterion. Retain
# the failed zero-spill target explicitly: the four extra accumulator registers
# exceed the main's168-register allocation. Only paired H200 time can decide
# whether removing scalar work pays for those spills. Do not report it as met.
combine=next(x for x in rr if 'vq8_combine' in x['symbol']);bcombine=next(x for x in br if 'vq8_combine' in x['symbol'])
assert {k:v for k,v in combine.items() if k!='symbol'}=={k:v for k,v in bcombine.items() if k!='symbol'},(combine,bcombine)
warnings=re.findall(r"\(C75\d+\).*?function '([^']+)'",log)
assert not any('vq8_main' in n for n in warnings),warnings
sass=subprocess.run(['cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
(native/'pvsum.sass').write_text(sass)
proof=dict(probes=rows,sm90a_compile=True,registers=re.findall(r'Used (\d+) registers',log),
    spill_frames=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log),
    main_resources=main,no_spill_target_met=no_spills,unchanged_combine=combine,baseline_combine=bcombine,c75_warnings=warnings,
    native_runtime=False,cuda_sha256=hashlib.sha256(m.VQ8_CUDA.encode()).hexdigest())
(out/'probe_native.json').write_text(json.dumps(proof,indent=2)+'\n');print(proof)
