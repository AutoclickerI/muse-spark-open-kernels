"""Compile original, oracle and phase-clock bodies; record actual resources and copy/MMA instructions."""
import hashlib,json,re,subprocess
from pathlib import Path
import source
H=Path(__file__).resolve().parent;(H/'out').mkdir(exist_ok=True)
rows=[]
for family in ('d','v'):
 for kind in ('A','M','CLOCK'):
    name=family+kind;src=source.original(family) if kind=='A' else source.source(family,clocks=kind=='CLOCK')
    p=H/'out'/(name+'.cu');p.write_text(src)
    r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True)
    log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log);assert r.returncode==0,(name,log[-3000:]);assert not re.search(r'C75\d+',log),(name,log[-2000:])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
    resources=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
    copies=len(re.findall(r'\bLDGSTS\b',sass));tensor=len(re.findall(r'\b[HQ]GMMA\b',sass));assert copies and tensor,(name,copies,tensor)
    rows.append(dict(name=name,sha256=hashlib.sha256(src.encode()).hexdigest(),resources=resources,copy_sites=copies,tensor_sites=tensor,no_C75=True))
    print('PASS',name,resources.replace('\n',' ')[-280:],flush=True)
(H/'native.json').write_text(json.dumps(rows,indent=2)+'\n')
