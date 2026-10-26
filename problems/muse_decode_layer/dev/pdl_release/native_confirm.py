"""Compile new combined VDSK release/wait specializations without launching."""
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
import torch
from triton.backends.compiler import GPUTarget
from triton.runtime import driver
H=Path(__file__).resolve().parent;out=H/'confirmation/native';out.mkdir(exist_ok=True)
sp=importlib.util.spec_from_file_location('release_confirm',H/'confirmation/C.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
old=driver.active.get_current_target;rows=[]
try:
 driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
 for chain in (False,True):
  M,N,K,S=128,4608 if chain else 2304,8192,3
  BT,BW,BK,NS,NW,TMA=(128,128,64,5,8,1) if chain else (128,64,64,8,4,1)
  x=torch.empty((M,K),device='cuda',dtype=torch.bfloat16);w=torch.empty((N,K),device='cuda',dtype=torch.bfloat16);p=torch.empty((S,M,N),device='cuda')
  xd=m._VDSK_TD.from_tensor(x,[BT,BK]);wd=m._VDSK_TD.from_tensor(w,[BW,BK])
  k=m._vdsk_gemm.warmup(x,w,p,xd,wd,M,N,K,S,BT=BT,BW=BW,BK=BK,NS=NS,TMA=TMA,CHAIN=chain,launch_pdl=True,num_warps=NW,num_stages=NS,grid=(N//BW,S))
  f=out/('chain'+str(int(chain))+'.cubin');f.write_bytes(k.asm['cubin'])
  sass=subprocess.run(['cuobjdump','--dump-sass',str(f)],capture_output=True,text=True,check=True).stdout
  ops=[l.strip() for l in sass.splitlines() if '/*' in l and ';' in l]
  release=[i for i,l in enumerate(ops) if 'PREEXIT' in l];wait=[i for i,l in enumerate(ops) if 'ACQBULK' in l];loads=[i for i,l in enumerate(ops) if re.search(r'\b(LDG|LDGSTS|UTMALDG)',l)]
  assert len(release)==1 and min(release)<min(loads),(release,loads[:1])
  assert (bool(wait)==chain) and (not chain or min(release)<min(wait)<min(loads)),(chain,release,wait,loads[:1])
  usage=subprocess.run(['cuobjdump','--dump-resource-usage',str(f)],capture_output=True,text=True,check=True).stdout
  rows.append(dict(chain=chain,ptx_sha256=hashlib.sha256(k.asm['ptx'].encode()).hexdigest(),release=release,wait=wait,first_load=loads[:1],resources=usage,shared=k.metadata.shared))
finally:driver.active.get_current_target=old
(H/'confirmation/native_compile.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
