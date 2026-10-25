"""Hopper resource and async-pipeline admission; no inferred throughput claim."""
import collections
import hashlib
import json
from pathlib import Path
import re
H=Path(__file__).resolve().parent
out={}
for mode,ns in (('transpose','rt'),('cta2','oc'),('cta2s3','oc')):
 p=Path('/tmp')/('p3b_r2_'+mode);log=p.with_name(p.name+'_ptxas.log').read_text();sass=p.with_suffix('.sass').read_text();rows=[]
 assert not re.search(r'C751[239]|C7520',log),(mode,'serialized WGMMA')
 for block in log.split('Compiling entry function ')[1:]:
  z=re.search(ns+r'6kernelILi(\d+)E(?:Li(\d+)E)?Lb(\d)',block.splitlines()[0])
  if not z:continue
  frame=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',block)
  assert frame and not any(map(int,frame.groups())),(mode,z.groups(),frame.groups())
  reg=int(re.search(r'Used (\d+) registers',block)[1]);rows.append(dict(parameters=z.groups(),registers=reg,stack_spill=list(map(int,frame.groups()))))
 kernels=[]
 for f in sass.split('Function : ')[1:]:
  if ns+'6kernel' not in f.splitlines()[0]:continue
  op=collections.Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]\d+\s+)?([A-Z][A-Z0-9_.]+)',f))
  assert not any(k.startswith(('LDL','STL')) for k in op),(mode,'local-memory SASS')
  dep=collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',f))
  assert dep.get('0x1',0)>0,(mode,'group1 missing')
  kernels.append(dict(name=f.splitlines()[0].strip(),counts=dict(op),group_wait=dict(dep)))
 assert rows and kernels
 out[mode]=dict(cuda_sha256=hashlib.sha256((H/(mode+'.cu')).read_bytes()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),resources=rows,kernels=kernels)
(H/'resources.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in out.items():print(k,[(r['parameters'],r['registers']) for r in v['resources']])
