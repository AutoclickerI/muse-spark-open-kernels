"""Native build and pre-wait instruction audit of every round-2 specialization."""
import hashlib,json,re,subprocess
from pathlib import Path
import kernels
H=Path(__file__).resolve().parent;rows=[]
for kind in ('last','early','dsm'):
    s=kernels.source(kind);p=H/'out'/(kind+'.cu');p.write_text(s)
    r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],text=True,capture_output=True)
    p.with_suffix('.compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(kind,(r.stdout+r.stderr)[-2000:])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
    res=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
    inventory=[]
    for block in sass.split('Function : ')[1:]:
        ins=[(int(m[1],16),m[2]) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\*',block)]
        waits=[pc for pc,op in ins if 'ACQBULK' in op]
        if not waits:continue
        before=[dict(pc=hex(pc),op=op) for pc,op in ins if pc<min(waits) and ('LDGSTS' in op or 'LDSM' in op)]
        if kind=='early':assert sum('LDGSTS' in x['op'] for x in before)==8 and sum('LDSM' in x['op'] for x in before)==4
        inventory.append(dict(symbol=block.splitlines()[0],waits=[hex(x) for x in waits],prewait=before))
    assert len(inventory)==8,(kind,len(inventory))
    rows.append(dict(kind=kind,cuda_sha256=hashlib.sha256(s.encode()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),resources=res,kernels=inventory))
(H/'out/native_admission.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS sm90a',[(r['kind'],len(r['kernels'])) for r in rows])
