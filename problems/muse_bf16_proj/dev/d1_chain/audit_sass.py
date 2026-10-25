"""sm90a ordering and resource proof for the actual decode-1 cluster configurations."""
import concurrent.futures,json,re,subprocess
from pathlib import Path
from audit_native import compile_one,load
from patch import source
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text();m=load(ROOT/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
folder=HERE/'native';folder.mkdir(exist_ok=True)
items=[('baseline',m.GLUE_CL_CUDA)]+[(n,source(base,n)) for n in ('weights','late_prew','hints','stream')]
with concurrent.futures.ThreadPoolExecutor(2) as pool:rows=list(pool.map(lambda item:compile_one(item,folder),items))
assert all(r['ok'] for r in rows)
out=[]
for row in rows:
 sass=(folder/(row['name']+'.sass')).read_text();usage=(folder/(row['name']+'.resources')).read_text()
 for kernel in row['kernels']:
  symbol=kernel['symbol']
  if not (('route_clILi8ELi256' in symbol) or ('boundary_clILi8ELi256ELi1' in symbol) or ('route_clILi4ELi256' in symbol) or ('boundary_clILi4ELi256ELi1' in symbol)):continue
  resource=next(b for b in usage.split('Function ')[1:] if symbol in b)
  block=next(b for b in sass.split('Function : ')[1:] if b.startswith(symbol))
  ins=[s.strip() for s in block.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s)]
  waits=[i for i,s in enumerate(ins) if 'ACQBULK' in s];assert waits
  before=[s for s in ins[:waits[0]] if re.search(r'\b(LDG|CCTL)\b',s)]
  regs=int(re.search(r'REG:(\d+)',resource)[1]);local=int(re.search(r'LOCAL:(\d+)',resource)[1])
  out.append(dict(mode=row['name'],symbol=symbol,registers=regs,local_bytes=local,waits=waits,pre_wait=before,source_sha256=row['sha256']))
  if row['name']=='baseline' and 'Li8E' in symbol:assert not before
  if row['name']!='baseline':assert before and local==0,(row['name'],symbol,regs,local)
(HERE/'sass_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS selected sm90a preloads precede wait; resource/spill counts retained',len(out))
