"""Build native fused specializations and audit their MMA grouping and local memory."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from fused import FT_CONFIGS

HERE=Path(__file__).resolve().parent


def main():
    full=(HERE/'fused.cu').read_text()
    src='#include <cuda_bf16.h>\n#include <stdint.h>\n'+full[full.index('namespace mwft'):full.index('#define FT_CASES')]+'\n}\n'
    for nt,kt,s,sk in FT_CONFIGS:
        for first in ('true','false'):
            src+=f'template __global__ void mwft::fused<{nt},{kt},{s},{sk},{first}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,const int*,const float*,const int*,const __nv_bfloat16*,float*,__nv_bfloat16*,float*,float*,float*,int*,__nv_bfloat16*,int,int,int);\n'
    path=HERE/'diagnostics/fused.cu';path.write_text(src)
    p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True)
    (HERE/'diagnostics/fused.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode()
    assert 'C7513' not in p.stderr
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
        waits=[arg for op,arg in ops if op.startswith('WARPGROUP.DEPBAR')]
        row=dict(name=name,registers=regs[name],local_memory=sum(op.startswith(('LDL','STL')) for op,_ in ops),
                 mma=sum(op.startswith('HGMMA.') and '.BF16' in op for op,_ in ops),waits=waits)
        assert row['local_memory']==0,row
        assert len(waits)==3 and waits[0].endswith('0x1') and waits[1].endswith('0x1') and waits[2].endswith('0x0'),row
        rows.append(row)
    assert len(rows)==2*len(FT_CONFIGS)
    report=dict(sha256=hashlib.sha256(full.encode()).hexdigest(),kernels=rows)
    (HERE/'diagnostics/fused_sass_counts.json').write_text(json.dumps(report,indent=2)+'\n')
    print('NATIVE_FUSED_PASS',len(rows),'registers',min(r['registers'] for r in rows),max(r['registers'] for r in rows),'zero local memory; wait1/wait1/final wait0')


if __name__=='__main__':main()
