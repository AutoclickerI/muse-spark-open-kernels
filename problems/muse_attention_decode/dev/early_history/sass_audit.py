"""Rebuild sm90a artifacts and retain bounded instruction-order/resource evidence."""
import hashlib,json,re,subprocess
from pathlib import Path
import history_build
HERE=Path(__file__).resolve().parent;OUT=HERE/'out'

def run(args):return subprocess.run(args,text=True,capture_output=True,check=True).stdout
rows=[]
for name,source in [('baseline',history_build.build_last.source()),('early',history_build.source()),('late',history_build.source(False)),('clock',history_build.source(clock=True))]:
    p=OUT/(name+'.cu');p.write_text(source)
    result=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True,check=True)
    p.with_suffix('.compile.log').write_text(result.stdout+result.stderr)
    sass=run(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))]);p.with_suffix('.sass').write_text(sass)
    usage=run(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))]);p.with_suffix('.resources').write_text(usage)
    kernels=[]
    for block in sass.split('Function : ')[1:]:
        symbol=block.splitlines()[0].strip()
        instructions=[]
        for line in block.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\*',line)
            if m:instructions.append((int(m[1],16),m[2]))
        waits=[pc for pc,op in instructions if 'ACQBULK' in op]
        if not waits:continue
        first=min(waits)
        copy=[dict(pc=hex(pc),op=op) for pc,op in instructions if 'LDGSTS' in op and pc<first]
        trans=[dict(pc=hex(pc),op=op) for pc,op in instructions if 'LDSM' in op and pc<first]
        assert copy if name!='late' else not copy
        if name in ('early','clock'):assert len(copy)==8 and len(trans)==4
        kernels.append(dict(symbol=symbol,waits=[hex(w) for w in waits],history_copy_before_first_wait=copy,v_transpose_before_first_wait=trans))
    rows.append(dict(name=name,cuda_sha256=hashlib.sha256(source.encode()).hexdigest(),sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),resource_usage=usage,kernels=kernels))
(HERE/'sass_proof.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',[(r['name'],len(r['kernels'])) for r in rows])
