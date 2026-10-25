"""Record real sm90a PV-probe instruction changes, not source-level estimates."""
from collections import Counter
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
H=Path(__file__).resolve().parent
rows=[]
with tempfile.TemporaryDirectory(prefix='raw-v-sass-') as tmp:
 for name in ['regv','ldsv']:
  p=H/(name+'.cu');obj=Path(tmp)/(name+'.cubin')
  subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-cubin',str(p),'-o',str(obj)],check=True,capture_output=True)
  sass=subprocess.run(['cuobjdump','--dump-sass',str(obj)],check=True,capture_output=True,text=True).stdout
  (H/(name+'.sass')).write_text(sass)
  body=next(x for x in sass.split('Function : ') if x.splitlines() and 'regv_probe' in x.splitlines()[0])
  ins=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)',body)
  rows.append(dict(source=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='standalone PV probe including same P packing and final O restore',instructions=len(ins),counts=dict(sorted(Counter(ins).items()))))
(H/'pv_sass.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row['source'],row['instructions'],{k:v for k,v in row['counts'].items() if k.startswith(('LDS','LDSM','SHFL','PRMT'))})
