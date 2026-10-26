"""Compare real instruction/resource changes; compile repaired producer-context clock bodies."""
import hashlib,json,re,subprocess
from pathlib import Path
import structural
H=Path(__file__).resolve().parent;(H/'out').mkdir(exist_ok=True);rows=[]
for f in ('d','v'):
 for k in ('EXP','CLOCK_BASE','CLOCK_EXP'):
    s=structural.source(f) if k=='EXP' else structural.context_clock(f,k=='CLOCK_EXP');p=H/'out'/(f+k+'.cu');p.write_text(s)
    r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True);log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log)
    assert r.returncode==0,(f,k,log[-2500:]);assert not re.search(r'C75\d+',log),(f,k,log[-1000:])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
    resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
    counts={n:len(re.findall(r'\b'+n+r'\b',sass)) for n in ('MUFU.EX2','F2I','FFMA','QGMMA','LDGSTS')}
    rows.append(dict(name=f+k,sha256=hashlib.sha256(s.encode()).hexdigest(),resources=resources,instruction_sites=counts,no_C75=True));print('PASS',f,k,counts,resources.replace('\n',' ')[-160:],flush=True)
(H/'structural_native.json').write_text(json.dumps(rows,indent=2)+'\n')
