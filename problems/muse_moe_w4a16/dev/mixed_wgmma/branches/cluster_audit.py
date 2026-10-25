"""Audit native decode-once Hopper resources and each KT128 loop.

Usage: nvcc -std=c++20 -O3 -lineinfo -gencode=arch=compute_90a,code=sm_90a
          -cubin cu_cluster.cu -o /tmp/p3b_cluster.cubin -Xptxas -v 2>ptxas.log
       cuobjdump -sass /tmp/p3b_cluster.cubin > /tmp/p3b_cluster.sass
       python3 cluster_audit.py ptxas.log /tmp/p3b_cluster.sass
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
H=Path(__file__).resolve().parent
log=Path(sys.argv[1]).read_text();rows=[]
for b in log.split('Compiling entry function ')[1:]:
 n=b.splitlines()[0];z=re.search(r'oc6kernelILi(\d+)ELb(\d)ELi(\d+)',n)
 if not z:continue
 sm=re.search(r'(\d+) bytes smem',b);fr=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',b)
 nt,f,cs=map(int,z.groups());st=2 if nt<=128 else 1
 rows.append(dict(maxnt=nt,first=f,cs=cs,registers=int(re.search(r'Used (\d+) registers',b)[1]),static_smem=int(sm[1]) if sm else 0,
                  dynamic_smem=((st*4608+1023)//1024)*1024+st*nt*256,stack_spill=list(map(int,fr.groups()))))
sha=hashlib.sha256((H/'cu_cluster.cu').read_bytes()).hexdigest()
(H/'cluster_resources.json').write_text(json.dumps(dict(cuda_sha256=sha,resources=rows),indent=2)+'\n')
s=Path(sys.argv[2]).read_text();out={}
for b in s.split('Function : ')[1:]:
 n=b.splitlines()[0];z=re.search(r'oc6kernelILi(\d+)ELb(\d)ELi(\d+)',n)
 if not z:continue
 ins=[(int(a,16),o) for a,o in re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',b)];loops=[]
 for pc,op in ins:
  p=re.search(r'BRA (0x[0-9a-f]+)',op)
  if p and int(p[1],16)<pc:
   lo=int(p[1],16);body=[x for t,x in ins if lo<=t<=pc];hg=[x for x in body if 'HGMMA' in x]
   if len(hg)==8:
    loops.append(dict(start=lo,end=pc,NT=int(re.search(r'HGMMA.64x(\d+)x',hg[0])[1]),KT128_instructions=len(body),HMUL2=sum('HMUL2' in x for x in body),LDGSTS=sum('LDGSTS' in x for x in body),TMA=sum('UTMALDG' in x for x in body),local=sum('LDL' in x or 'STL' in x for x in body)))
 loops=[x for x in loops if x['KT128_instructions']==min(y['KT128_instructions'] for y in loops if y['NT']==x['NT'])]
 out['/'.join(z.groups())]=dict(depbar=dict(collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',b))),loops=loops)
(H/'cluster_sass.json').write_text(json.dumps(dict(cuda_sha256=sha,sass_sha256=hashlib.sha256(s.encode()).hexdigest(),kernels=out),indent=2)+'\n')
print('REGS',[(r['maxnt'],r['first'],r['cs'],r['registers'],r['stack_spill']) for r in rows]);print('LOOPS',[(k,[r['KT128_instructions'] for r in v['loops']]) for k,v in out.items() if k.endswith('/4')])
