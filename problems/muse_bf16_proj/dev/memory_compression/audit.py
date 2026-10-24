import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];out=ROOT/'runs/screen'
obj=out/'memory_compression90.o';src=HERE/'kernel.cu'
p=subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17','-O3','--ptxas-options=-v',
    '-c',str(src),'-o',str(obj)],capture_output=True,text=True,check=True)
(out/'memory_compression90.compile').write_text(p.stderr)
spills=[tuple(map(int,s)) for s in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)]
assert len(spills)==3 and not any(a or b for a,b in spills),spills
sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'memory_compression90.sass').write_text(sass)
assert len(sass.split('Function : '))-1==3
assert all('LDG.E.128' in s for s in sass.split('Function : ')[1:])
payload=ROOT/'prof/lab/generic_memory_compression.py'
spec=importlib.util.spec_from_file_location('mc_audit',payload);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for n in (8,64):
    kernel=triton.compile(ASTSource(m.mc_ordered_finish,
        dict(Y='*fp32',IDS='*i32',WTS='*fp32',PNW='*bf16',OUT='*bf16'),
        constexprs=dict(NP=n,PAIRS=n)),target=GPUTarget('cuda',90,32),options=dict(num_warps=8))
    (out/f'mc_ordered_np{n}.ptx').write_text(kernel.asm['ptx'])
report=dict(payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
    kernel_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),native_kernels=3,
    triton_ordered_controls=2,no_native_spills=True,all_stream_bytes_use_volatile_vector_loads=True)
(HERE/'compile_sm90a.json').write_text(json.dumps(report,indent=2)+'\n')
print('MC_SM90A',json.dumps(report),flush=True)
