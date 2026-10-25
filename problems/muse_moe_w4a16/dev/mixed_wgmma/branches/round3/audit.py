"""Verify unchanged control prefix and new Hopper resource/pipeline admission."""
import collections
import hashlib
import json
from pathlib import Path
import re
H=Path(__file__).resolve().parent
cu=(H/'tail.cu').read_text();ctrl=(H.parent/'round2/cta2.cu').read_text();assert cu.startswith(ctrl)
log=Path('/tmp/p3b_r3_tail_ptxas.log').read_text();sass=Path('/tmp/p3b_r3_tail.sass').read_text();rows=[];kernels=[]
assert not re.search(r'C751[239]|C7520',log),'serialized WGMMA'
for block in log.split('Compiling entry function ')[1:]:
 z=re.search(r'tail6kernelILi(\d+)ELi(\d+)ELb(\d)',block.splitlines()[0])
 if not z:continue
 frame=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block);assert frame and not any(map(int,frame.groups()))
 rows.append(dict(mode=int(z[1]),stages=int(z[2]),first=int(z[3]),registers=int(re.search(r'Used (\d+) registers',block)[1]),stack_spill=list(map(int,frame.groups()))))
for f in sass.split('Function : ')[1:]:
 if 'tail6kernel' not in f.splitlines()[0]:continue
 op=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]+)',f));assert not any(k.startswith(('LDL','STL')) for k in op)
 dep=collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',f));assert dep.get('0x1',0)>0
 kernels.append(dict(name=f.splitlines()[0].strip(),instructions=dict(op),group_wait=dict(dep)))
assert len(rows)==len(kernels)==6
out=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),CTA2_prefix_sha256=hashlib.sha256(ctrl.encode()).hexdigest(),control_prefix_equal=True,sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),resources=rows,kernels=kernels)
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n');print(rows)
