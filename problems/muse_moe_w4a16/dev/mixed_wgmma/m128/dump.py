"""Compile the actual M128 Hopper specializations and record resource use."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from source import source,CONFIGS

HERE=Path(__file__).resolve().parent


def main():
    full=source();s='#include <cuda_bf16.h>\n#include <stdint.h>\n'+full[full.index('namespace mw128'):full.index('#define FT_CASES')]+'\n}\n'
    for nt,kt,st,sk,ad in CONFIGS:
        for first in ('true','false'):
            s+=f'template __global__ void mw128::fused<{nt},{kt},{st},{sk},{str(ad).lower()},{first}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int);\n'
    path=HERE/'native.cu';path.write_text(s)
    p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True)
    path.with_suffix('.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode()
    assert 'C7513' not in p.stderr,p.stderr
    sass=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],capture_output=True,text=True,check=True).stdout
    path.with_suffix('.sass').write_text(sass);regs={};name=None
    for line in p.stderr.splitlines():
        if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
        m=re.search(r'Used (\d+) registers',line)
        if m:regs[name]=int(m[1])
    rows=[]
    for part in sass.split('Function : ')[1:]:
        name=part.splitlines()[0].strip()
        if 'fused' not in name:continue
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part)
        row=dict(name=name,registers=regs[name],local_memory=sum(op.startswith(('LDL','STL')) for op,_ in ops),mma=sum(op.startswith('HGMMA.') for op,_ in ops),waits=[arg for op,arg in ops if op.startswith('WARPGROUP.DEPBAR')])
        assert row['local_memory']==0,row
        rows.append(row)
    assert len(rows)==2*len(CONFIGS),len(rows)
    (HERE/'native.json').write_text(json.dumps(dict(sha256=hashlib.sha256(full.encode()).hexdigest(),kernels=rows),indent=2)+'\n')
    print('M128_NATIVE_PASS',len(rows),'registers',min(r['registers'] for r in rows),max(r['registers'] for r in rows),'zero spills',flush=True)


if __name__=='__main__':main()
