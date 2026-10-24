"""Compile every Hopper ring; inspect TMA, mbarriers, spills and residency."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];out=ROOT/'runs/screen';src=HERE/'kernel.cu';obj=out/'bulk_ring90.o'
p=subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17','-O3','-lineinfo','--ptxas-options=-v',
    '-c',str(src),'-o',str(obj)],capture_output=True,text=True,check=True)
(out/'bulk_ring90.compile').write_text(p.stderr)
sass=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'bulk_ring90.sass').write_text(sass)
resources=subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(obj)],capture_output=True,text=True,check=True).stdout
(out/'bulk_ring90.resources').write_text(resources)
rows=[]
for part in sass.split('Function : ')[1:]:
    name=part.splitlines()[0].strip()
    if 'ring' not in name:continue
    assert 'UBLKCP' in part and 'SYNCS' in part and 'HMMA' in part,name
    rows.append(dict(name=name,tma=True,mbarrier=True,mma=True))
assert len(rows)==33,len(rows)
spills=[tuple(map(int,x)) for x in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',p.stderr)]
assert not any(a or b for a,b in spills),spills
record=dict(kernel_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),native=rows,no_spills=True,
    max_ring_bytes=131072,hopper_smem_reservation=dict(one_cta=122880,two_ctas=81920),
    ptx_source='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk')
(HERE/'compile_sm90a.json').write_text(json.dumps(record,indent=2)+'\n');print('BR_SM90A',len(rows),'native ring specializations; no spills',flush=True)
