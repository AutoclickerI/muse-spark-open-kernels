"""Hopper native resource/order audit and diagnostic Triton cross-compilation."""
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
src=HERE/'kernel.cu';obj=out/'gemv_fixed90.o'
p=subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17','-O3',
    '-lineinfo','--ptxas-options=-v','-c',str(src),'-o',str(obj)],capture_output=True,text=True,check=True)
(out/'gemv_fixed90.compile').write_text(p.stderr)
ss=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'gemv_fixed90.sass').write_text(ss)
usage=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'gemv_fixed90.resources').write_text(usage)
native=[]
for part in ss.split('Function : ')[1:]:
    name=part.splitlines()[0].strip()
    if 'direct' not in name:continue
    trace='Lb1E' in name
    assert 'ACQBULK' in part and part.index('LDG.E')<part.index('ACQBULK'),name
    if trace:assert 'GLOBALTIMER' in part and ('CLOCK' in part or 'S2R' in part),name
    native.append(dict(name=name,trace=trace,weight_before_wait=True))
assert len(native)==28,len(native)
spills=[tuple(map(int,s)) for s in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)]
assert not any(a or b for a,b in spills),spills
payload=ROOT/'prof/lab/dense_gemv_fixed.py'
spec=importlib.util.spec_from_file_location('gf_audit',payload);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
def compile_one(name,fn,sig,const,warps):
    kernel=triton.compile(ASTSource(fn,sig,constexprs=const),target=GPUTarget('cuda',90,32),
        options=dict(num_warps=warps,num_stages=4,launch_pdl=True))
    (out/(name+'.ptx')).write_text(kernel.asm['ptx'])
    assert '%clock64' in kernel.asm['ptx'] and '%globaltimer' in kernel.asm['ptx']
    rows.append(dict(name=name,shared=kernel.metadata.shared))
for t in (1,8,16,32):
    bn,ks=(32,1024) if t<=16 else (64,512)
    compile_one(f'gf_gemv_t{t}',m.gd_gemv_kernel,
        dict(STAMPS='*i64',X='*bf16',W='*bf16',P='*fp32',T='i32',x_stride='i32'),
        dict(N=4608,K=8192,KS=ks,BN=bn,BK=128,TB=max(16,t),PREFETCH=False,USE_PDL=True),4)
    for splits in (1,8):
        compile_one(f'gf_boundary_t{t}_s{splits}',m.gd_boundary_fast,
            dict(STAMPS='*i64',BR='*bf16' if splits==1 else '*fp32',STREAM='*fp32',SOUT='*fp32',OUT='*bf16',
                 POSTW='*fp32',ALPHA='*fp32',BETA='*fp32',PREW='*fp32'),
            dict(S=splits,BS=t*8192,RS=8192,NB=8192,HAS_POSTW=False,REPLICAS=1,USE_PDL=True),8)
        compile_one(f'gf_route_t{t}_s{splits}',m.gd_route_fast,
            dict(STAMPS='*i64',PR='*fp32',BIAS='*fp32',PREW='*fp32',LSTART='*i32',X='*bf16',IDS='*i32',WTS='*fp32',LIDS='*i32'),
            dict(S=splits,PS=t*4608,RS=4608,REPLICAS=1,USE_PDL=True),8)
        compile_one(f'gf_qk_t{t}_s{splits}',m.gd_qk_prep_kernel,
            dict(STAMPS='*i64',P='*bf16' if splits==1 else '*fp32',ps_stride='i32',POS='*i64',LOC='*i64',
                 KC='*fp8e4nv',VC='*fp8e4nv',KSCALE='*fp32',VSCALE='*fp32',INV_FREQ='*fp32',QOUT='*fp8e4nv',GATE='*fp32',q_mult='fp32',eps='fp32'),
            dict(S=splits,HQ=16,HKV=2,D=64,ROPE=False,Q_FP8=True,USE_PDL=True),4)
report=dict(kernel_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
            native=native,triton=rows,no_native_spills=True)
(HERE/'compile_sm90a.json').write_text(json.dumps(report,indent=2)+'\n')
print('GF_SM90A',len(native),'native',len(rows),'Triton diagnostics; initial weight LDG before ACQBULK; no native spills',flush=True)
