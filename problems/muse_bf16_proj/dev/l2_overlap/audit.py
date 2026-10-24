"""Compile the actual byte-pointer transport for H200; retain SASS and provenance."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
src=HERE/'prefetch.cu';out=ROOT/'runs/screen/p7_l2.o'
compiled=subprocess.run([str(ROOT/'tools/img'),'/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17','-O3',
    '-lineinfo','--ptxas-options=-v','-c',str(src),'-o',str(out)],capture_output=True,text=True,check=True)
sass=subprocess.run([str(ROOT/'tools/img'),'/usr/local/cuda/bin/cuobjdump','--dump-sass',str(out)],capture_output=True,text=True,check=True).stdout
(ROOT/'runs/screen/p7_l2.sass').write_text(sass)
spills=[tuple(map(int,s)) for s in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',compiled.stderr)]
assert spills and not any(a or b for a,b in spills)
assert 'LDG.E.128' in sass
assert 'UBLKPF.L2' in sass,sass[-1000:]
report=dict(cuda_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),arch='sm_90a',no_spills=True,
    registers=list(map(int,re.findall(r'Used (\d+) registers',compiled.stderr))),
    native_loads_128=sass.count('LDG.E.128'),native_bulk_instructions=sass.count('UBLKPF.L2'))
(HERE/'compile_sm90a.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
