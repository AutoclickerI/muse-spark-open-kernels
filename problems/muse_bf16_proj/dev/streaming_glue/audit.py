"""Compile the exact fused source for sm_90a and retain native load/MMA evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
payload=HERE.parent/'streaming_glue_lab.py'
tree=ast.parse(payload.read_text())
src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
         and any(isinstance(t,ast.Name) and t.id=='SP_CUDA' for t in n.targets))
tag=hashlib.sha256(src.encode()).hexdigest()[:16]
scratch=REPO/'runs/screen/streaming_glue';scratch.mkdir(parents=True,exist_ok=True)
cu,obj=scratch/f'{tag}.cu',scratch/f'{tag}.o';cu.write_text(src)
compiled=subprocess.run([str(REPO/'tools/img'),'/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++17',
                         '-O3','-lineinfo','--ptxas-options=-v','-c',str(cu),'-o',str(obj)],
                        capture_output=True,text=True,check=True)
(scratch/f'{tag}.ptxas.log').write_text(compiled.stderr)
sass=subprocess.run([str(REPO/'tools/img'),'/usr/local/cuda/bin/cuobjdump','--dump-sass',str(obj)],
                    capture_output=True,text=True,check=True).stdout
functions=[]
for block in re.split(r'ptxas info    : Compiling entry function ',compiled.stderr)[1:]:
    name=re.search(r"'([^']+)'",block)[1];regs=int(re.search(r'Used (\d+) registers',block)[1])
    stores,loads=map(int,re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads',block).groups())
    functions.append(dict(name=name,registers=regs,spill_stores=stores,spill_loads=loads))
counts={op:len(re.findall(op,sass)) for op in
        (r'LDG\.E\.128',r'LDGSTS\.E\.BYPASS\.128',r'HMMA\.16816\.F32\.BF16',r'ATOM',r'MEMBAR')}
out=dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
         arch='sm_90a',kernels=functions,sass_counts=counts,
         all_function_spills=[list(map(int,m)) for m in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',compiled.stderr)])
(HERE/'compile_sm90a.json').write_text(json.dumps(out,indent=2)+'\n')
if len(functions)!=33:raise RuntimeError(f'expected 32 projection kernels + read control, got {len(functions)}')
if any(f['spill_stores'] or f['spill_loads'] for f in functions):
    raise RuntimeError('native spilling: '+str([f for f in functions if f['spill_stores'] or f['spill_loads']]))
if any(a or b for a,b in out['all_function_spills']):raise RuntimeError('spill in a device continuation')
if not all(counts.values()):raise RuntimeError(f'missing native instruction family: {counts}')
print(json.dumps(dict(kernels=len(functions),max_registers=max(f['registers'] for f in functions),no_spills=True,sass_counts=counts)))
