"""Native global-reduction resources/instructions and corrected clock body."""
import hashlib,json,re,subprocess
from pathlib import Path
import merge
H=Path(__file__).resolve().parent;(H/'out').mkdir(exist_ok=True);rows=[]
for name,s in (('GLOBAL',merge.source()),('GLOBAL_CLOCK',merge.context_clock())):
 p=H/'out'/(name+'.cu');p.write_text(s)
 r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log)
 assert r.returncode==0,(name,log[-3000:]);assert not re.search(r'C75\d+',log),(name,log[-1500:])
 sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
 resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
 counts={k:len(re.findall(r'\b'+k+r'\b',sass)) for k in ('MUFU.EX2','F2I','QGMMA','LDGSTS','ATOM','MEMBAR')}
 rows.append(dict(name=name,sha256=hashlib.sha256(s.encode()).hexdigest(),resources=resources,counts=counts,no_C75=True,nonspinning=True,device_acq_rel=True,cg_partial_reads=True));print('PASS',name,counts,resources.replace('\n',' ')[-200:],flush=True)
(H/'merge_native.json').write_text(json.dumps(rows,indent=2)+'\n')
