"""Strict complete-unit sm90a compile proofs, in bounded parallelism, for plain and clocks."""
import concurrent.futures
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path
H=Path(__file__).resolve().parent
manifest=json.loads((H/'manifest.json').read_text())
def check(row,clock):
 name=row['name'];p=H/(name+'.cu');define=('DECODE_CLOCK' if row['kind']=='attn_main' else 'VQ8_CLK')+'='+str(clock)
 with tempfile.TemporaryDirectory(prefix='decode-native-') as tmp:
  r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-D'+define,'-c',str(p),'-o',str(Path(tmp)/'unit.o')],capture_output=True,text=True)
  assert r.returncode==0,(name,clock,r.stderr)
  assert not re.search(r'C75\d+',r.stderr),(name,clock,r.stderr)
  assert all(int(a)==int(b)==0 for a,b in re.findall(r'(\d+) bytes spill stores, (\d+) bytes spill loads',r.stderr)),(name,clock,r.stderr)
  return dict(name=name,clock=clock,define=define,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),ok=True,log=r.stderr)
with concurrent.futures.ThreadPoolExecutor(4) as pool:
 futures=[pool.submit(check,row,c) for row in manifest['variants'] for c in (0,1)]
 rows=[]
 for f in futures:
  row=f.result();rows.append(row);print('NATIVE PASS',row['name'],row['clock'],flush=True)
(H/'native.json').write_text(json.dumps(dict(passed=True,units=rows),indent=2)+'\n')
