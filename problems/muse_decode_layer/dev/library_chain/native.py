"""Compile the exact new projection specializations for Hopper without launch."""
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import subprocess
import sysconfig
from pathlib import Path
import torch
from torch.utils import cpp_extension
from triton.backends.compiler import GPUTarget
from triton.runtime import driver

H=Path(__file__).resolve().parent;out=H/'out/native';out.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('library_native',H.parents[1]/'cand_r3_best7_s15L.py')
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ns=m._nf_vendor();units=[]
class Mock:
    def __getattr__(self,name):return lambda *a,**kw:None
def capture(name,cpp_sources='',cuda_sources='',**kw):
    units.append((name,cuda_sources));return Mock()
ns['load_inline']=capture
ns['OLAYER_CUDA']=m._nf_pdl_wide(ns['OLAYER_CUDA'])
ns['build_olayer']([(128,128,4,3,1,1)])
ns['tt_v10_tune_RL_CUDA']=m._nf_pdl_wide(ns['tt_v10_tune_RL_CUDA'])
ns['tt_v10_tune_build_rl']([(256,64,5,7,1,1,1)])
base=['-std=c++20','-O3','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr',
      '-DTORCH_EXTENSION_NAME=library_native','-DTORCH_API_INCLUDE_EXTENSION_H','-Xptxas=-v',
      '-isystem',sysconfig.get_paths()['include']]
base += [a for p in cpp_extension.include_paths(device_type='cuda') for a in ('-isystem',p)]
def one(item):
    name,src=item;name=re.sub(r'[^a-zA-Z0-9_]','_',name);f=out/(name+'.cu');f.write_text(src)
    p=subprocess.run(['nvcc',*base,'-c',str(f),'-o',str(f.with_suffix('.o'))],capture_output=True,text=True,timeout=900)
    log=p.stdout+p.stderr;f.with_suffix('.log').write_text(log)
    row=dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),ok=p.returncode==0,
        registers=[int(x) for x in re.findall(r'Used (\d+) registers',log)],
        stack_spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log))
    if p.returncode:row['error']=log[-2500:];return row
    sass=subprocess.run(['cuobjdump','--dump-sass',str(f.with_suffix('.o'))],capture_output=True,text=True,check=True).stdout
    f.with_suffix('.sass').write_text(sass)
    row['release']='PREEXIT' in sass;row['wait']='ACQBULK' in sass
    assert row['release'] and row['wait'],name
    return row
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(one,units))
old=driver.active.get_current_target
try:
    driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
    for role,(tag,cfg,S) in m.LIBRARY_NATIVE_ROWS.items():
        M,name=role;N,K=(4608,8192) if name=='pool_router' else (8192,4096)
        BT,BW,BK,NS,NW,TMA=cfg
        x=torch.empty((M,K),device='cuda',dtype=torch.bfloat16)
        w=torch.empty((N,K),device='cuda',dtype=torch.bfloat16);p=torch.empty((S,M,N),device='cuda')
        xd=m._VDSK_TD.from_tensor(x,[BT,BK]) if TMA else None
        wd=m._VDSK_TD.from_tensor(w,[BW,BK]) if TMA else None
        k=m._vdsk_gemm.warmup(x,w,p,xd,wd,M,N,K,S,BT=BT,BW=BW,BK=BK,NS=NS,TMA=TMA,
            CHAIN=True,launch_pdl=True,num_warps=NW,num_stages=NS,grid=(M//BT*N//BW,S))
        (out/(name+'.ptx')).write_text(k.asm['ptx'])
        (out/(name+'.cubin')).write_bytes(k.asm['cubin'])
        sass=subprocess.run(['cuobjdump','--dump-sass',str(out/(name+'.cubin'))],capture_output=True,text=True,check=True).stdout
        (out/(name+'.sass')).write_text(sass)
        ops=[l.strip() for l in sass.splitlines() if '/*' in l and ';' in l]
        release=[i for i,l in enumerate(ops) if 'PREEXIT' in l];wait=[i for i,l in enumerate(ops) if 'ACQBULK' in l]
        loads=[i for i,l in enumerate(ops) if re.search(r'\b(LDG|LDGSTS|UTMALDG)',l)]
        assert release and wait and min(release)<min(loads) and min(wait)<min(loads),(name,release,wait,loads[:1])
        usage=subprocess.run(['cuobjdump','--dump-resource-usage',str(out/(name+'.cubin'))],capture_output=True,text=True,check=True).stdout
        rows.append(dict(name=name,ok=True,sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest(),
            shared=k.metadata.shared,resource_usage=usage,release_instruction=release,wait_instruction=wait,
            first_global_load=loads[:1],metadata=str(k.metadata)))
        del x,w,p
finally:driver.active.get_current_target=old
(H/'native_compile.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r,flush=True)
assert len(rows)==4 and all(r['ok'] for r in rows)
