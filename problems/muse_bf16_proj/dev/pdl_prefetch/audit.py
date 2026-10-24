"""Cross-compile every native path; verify actual weight operations precede ACQBULK."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
out=ROOT/'runs/screen';src=HERE/'kernel.cu';obj=out/'pdl_prefetch90.o'
compiled=subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17','-O3',
    '-lineinfo','--ptxas-options=-v','-c',str(src),'-o',str(obj)],capture_output=True,text=True,check=True)
(out/'pdl_prefetch90.compile').write_text(compiled.stderr)
sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'pdl_prefetch90.sass').write_text(sass)
spills=[tuple(map(int,s)) for s in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',compiled.stderr)]
assert len(spills)==17 and not any(a or b for a,b in spills)
native=[]
for part in sass.split('Function : ')[1:]:
    name=part.splitlines()[0].strip()
    if 'read_stream' in name:continue
    flags=re.findall(r'ELb([01])',name)
    early=flags[0]=='1'
    op='UBLKCP.S.G' if 'bulk_ring' in name else 'LDG.E.128'
    first=part.index(op);wait=part.index('ACQBULK')
    assert (first<wait)==early,(name,op,first,wait)
    native.append(dict(name=name,early=early,first_weight_instruction=op,weight_before_wait=first<wait))
assert len(native)==16
payload=ROOT/'prof/lab/dense_pdl_prefetch.py'
spec=importlib.util.spec_from_file_location('pp_audit',payload);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[]
for tokens,bn,ks in ((1,32,1024),(8,32,1024),(16,32,1024),(32,64,512)):
    for depth in (0,1,4):
        const=dict(N=4608,K=8192,KS=ks,BN=bn,BK=128,TB=max(16,tokens),DEPTH=depth,USE_PDL=True)
        sig=dict(X='*bf16',W='*bf16',P='*fp32',T='i32',x_stride='i32')
        kernel=triton.compile(ASTSource(m.pp_hint_gemv,sig,constexprs=const),target=GPUTarget('cuda',90,32),
            options={'num_warps':4,'num_stages':4,'launch_pdl':True})
        ptx=kernel.asm['ptx'];label=f'pdl_hint_t{tokens}_d{depth}'
        (out/(label+'.ptx')).write_text(ptx);(out/(label+'.cubin')).write_bytes(kernel.asm['cubin'])
        ss=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out/(label+'.cubin'))],
            capture_output=True,text=True,check=True).stdout
        assert ptx.count('griddepcontrol.wait')==1 and 'ACQBULK' in ss
        if depth:
            assert ptx.index('cp.async.bulk.prefetch')<ptx.index('griddepcontrol.wait')
            assert ss.index('UBLKPF.L2')<ss.index('ACQBULK')
        usage=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(out/(label+'.cubin'))],
            capture_output=True,text=True,check=True).stdout
        registers=int(re.search(r'REG:(\d+)',usage).group(1))
        local=int(re.search(r'LOCAL:(\d+)',usage).group(1))
        rows.append(dict(tokens=tokens,depth=depth,registers=registers,local_bytes=local,
                         shared_bytes=kernel.metadata.shared,weight_bulk_before_wait=bool(depth)))
        assert local==0
for splits in (1,2,8,16):
    kernel=triton.compile(ASTSource(m.pp_bf16_boundary,dict(P='*fp32',Y='*bf16'),
        constexprs=dict(S=splits,COUNT=32*8192,B=256,USE_PDL=True)),target=GPUTarget('cuda',90,32),
        options={'num_warps':4,'launch_pdl':True})
    ptx=kernel.asm['ptx'];assert ptx.index('griddepcontrol.wait')<ptx.index('ld.global')
    (out/f'pdl_boundary_s{splits}.ptx').write_text(ptx)
    rows.append(dict(kind='bf16_boundary',splits=splits,shared_bytes=kernel.metadata.shared))
report=dict(kernel_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
    payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),native=native,triton=rows,
    no_spills=True,registers=list(map(int,re.findall(r'Used (\d+) registers',compiled.stderr))),
    ptx_reference='https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html')
(HERE/'compile_sm90a.json').write_text(json.dumps(report,indent=2)+'\n')
print('PDL_SM90A_AUDIT',len(native),'native +12 Triton hints +4 BF16 boundaries; weight operations before dependency wait; no native/hint spills')
