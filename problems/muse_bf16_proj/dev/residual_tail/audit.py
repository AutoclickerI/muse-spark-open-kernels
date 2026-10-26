"""Compile actual sm90a source; retain ordered dependency loads and resources."""
import hashlib,json,pathlib,re,subprocess
H=pathlib.Path(__file__).resolve().parent;d=H/'native';d.mkdir(exist_ok=True);s=(H/'native_generated.py.inc').read_text();p=d/'residual.cu';p.write_text(s)
subprocess.run(['nvcc','-std=c++20','-O3','--expt-relaxed-constexpr','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-c',str(p),'-o',str(p.with_suffix('.o'))],check=True)
sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);usage=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass);p.with_suffix('.resources').write_text(usage);rows=[]
for block in sass.split('Function : ')[1:]:
 symbol=block.splitlines()[0];ins=[l.strip() for l in block.splitlines() if re.search(r'/\*[0-9a-f]+\*/',l)];waits=[i for i,l in enumerate(ins) if 'ACQBULK' in l]
 if not waits:continue
 assert not any(re.search(r'\b(LDL|STL)\b',l) for l in ins),symbol
 assert not any('LDG.' in l and 'STRONG.GPU' in l for l in ins[:waits[0]]),('dependent CG load before wait',symbol)
 rows.append(dict(symbol=symbol,waits=waits,ordered_loads_after_wait=sum('LDG.' in l and 'STRONG.GPU' in l for l in ins[waits[0]:]),stack_or_local_sass=False))
assert len(rows)==6,len(rows)
(H/'native_audit.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(s.encode()).hexdigest(),kernels=rows,resources=usage),indent=2)+'\n');print('PASS sm90a six boundary specializations, no stack/local SASS; ordered loads after wait')
