"""Compile actual new CUDA for sm90a and retain ordering/resources, not inferred runtime wins."""
import hashlib,json,pathlib,re,subprocess
H=pathlib.Path(__file__).resolve().parent;out=H/'native';out.mkdir(exist_ok=True);rows={}
for name in ('native','pmz'):
 src=(H/(name+'_generated.py.inc')).read_text();p=out/(name+'.cu');p.write_text(src);obj=p.with_suffix('.o')
 subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(obj)],check=True)
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(obj)],text=True);usage=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(obj)],text=True);p.with_suffix('.sass').write_text(sass);p.with_suffix('.resources').write_text(usage)
 kernels=[]
 for block in sass.split('Function : ')[1:]:
  symbol=block.splitlines()[0];ins=[l.strip() for l in block.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)];waits=[i for i,l in enumerate(ins) if 'ACQBULK' in l];releases=[i for i,l in enumerate(ins) if 'PREEXIT' in l]
  assert not any(re.search(r'\b(?:LDL|STL)\b',l) for l in ins),symbol
  before=[l for i,l in enumerate(ins) if waits and i<waits[0] and 'LDG.' in l]
  if name=='native' and waits:assert before and all('STRONG.SM' in l for l in before),('dependent load moved before wait',symbol,before)
  if waits or releases:kernels.append(dict(symbol=symbol,waits=waits,releases=releases,before_wait=before))
 assert kernels,(name,'no expected PDL instructions')
 rows[name]=dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),kernels=kernels,resources=usage)
 print('PASS sm90a',name,len(kernels),'PDL specializations, no local SASS',flush=True)
(H/'native_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
