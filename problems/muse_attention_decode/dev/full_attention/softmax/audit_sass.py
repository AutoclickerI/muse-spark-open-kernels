"""Count the isolated actual softmax/rescale probe and its source phases on sm90a."""
import argparse
from collections import Counter
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('names',nargs='*',default=['ref','mix25','mix50','gap','mixgap','tc25']);a=ap.parse_args()
rows=[]
with tempfile.TemporaryDirectory(prefix='softmax-sass-') as tmp:
 for name in a.names:
  p=H/(name+'.cu');cu=p.read_text();obj=Path(tmp)/(name+'.cubin')
  r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-cubin',str(p),'-o',str(obj)],capture_output=True,text=True)
  assert r.returncode==0,r.stderr
  sass=subprocess.run(['nvdisasm','-g',str(obj)],check=True,capture_output=True,text=True).stdout
  (H/(name+'.sass')).write_text(sass)
  # Restrict to the exported probe's text section; main has repeated unrolled steps.
  head=sass.index('\t.section\t.text._Z8sm_probe');tail=sass.find('\t.section',head+15)
  body=sass[head:tail if tail!=-1 else None]
  line=0;counts=Counter();phases={k:Counter() for k in ['max','alpha_sum','exp','pack','rescale','poly','tc_sum','other']}
  lines=cu.splitlines();index=lambda t:next(i+1 for i,x in enumerate(lines) if t in x)
  ranges=[('poly',index('DI float sm_poly'),index('return __uint_as_float(out)')+1),
          ('max',index('DI void softmax('),index('const float mn')-1),
          ('exp',index('CLK_MARK(se)'),index('CLK_ADD(0,66,se)')),
          ('alpha_sum',index('const float mn'),index('CLK_MARK(se)')-1),
          ('alpha_sum',index('CLK_MARK(sl)'),index('CLK_ADD(0,67,sl)')),
          ('pack',index('CLK_MARK(sp)'),index('CLK_ADD(0,68,sp)')),
          ('rescale',index('DI void rescale'),index('CLK_ADD(0,71,sr)'))]
  if name=='tc25':ranges.insert(0,('tc_sum',index('DI void sm_tc_sum'),index('l[0]+=d[0]')+1))
  for text in body.splitlines():
   hit=re.search(r'line (\d+)',text)
   if hit:line=int(hit.group(1))
   hit=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP]+\d+\s+)?([A-Z][A-Za-z0-9_.]+)',text)
   if not hit:continue
   op=hit.group(1);counts[op]+=1
   phase=next((n for n,lo,hi in ranges if lo<=line<=hi),'other');phases[phase][op]+=1
  row=dict(name=name,cuda_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='one 128-thread M64N64 probe: one softmax tile, P packing, complete O rescale; excludes QK/PV and producer',instructions=sum(counts.values()),counts=dict(counts),phases={k:dict(v) for k,v in phases.items()},resources=r.stderr)
  assert row['instructions']>50,(name,body[:1000])
  rows.append(row);print(name,row['instructions'],{k:v for k,v in counts.items() if k.startswith(('MUFU','FFMA','FADD','FMNMX','SHFL','F2FP','F2I','I2F','HGMMA'))},flush=True)
(H/'sass.json').write_text(json.dumps(rows,indent=2)+'\n')
