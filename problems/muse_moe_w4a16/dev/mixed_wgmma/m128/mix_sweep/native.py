"""Compile threshold consumers and N32/N64 heavy tiles on actual SM90a."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    result={}
    for p in ('swdist','sw_nt','sw_meta'):
        src=HERE/(p+'.cu');cube=HERE/(p+'.cubin')
        run=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(src),'-o',str(cube)],capture_output=True,text=True)
        (HERE/(p+'.ptxas.log')).write_text(run.stdout+run.stderr);run.check_returncode();assert 'C7513' not in run.stderr
        sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],check=True,capture_output=True,text=True).stdout
        (HERE/(p+'.sass')).write_text(sass);regs={};name=None
        for line in run.stderr.splitlines():
            if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
            m=re.search(r'Used (\d+) registers',line)
            if m:regs[name]=int(m[1])
        rows=[]
        for part in sass.split('Function : ')[1:]:
            name=part.splitlines()[0].strip();local=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?(?:LDL|STL)[.\s]',part);assert not local,(name,len(local))
            mma=re.findall(r'HGMMA\.64x(\d+)x16',part)
            if 'fusedILi64E' in name:assert mma and set(mma)=={'64'},(name,set(mma))
            rows.append(dict(name=name,registers=regs[name],local_memory_instructions=len(local),MMA_N=sorted(set(mma))))
        result[p]=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),kernels=rows)
    report=dict(entries=sum(len(v['kernels']) for v in result.values()),no_spills=True,sources=result)
    (HERE/'native.json').write_text(json.dumps(report,indent=2)+'\n');print('SWEEP_NATIVE_PASS',report['entries'],'entries, N32/N64 verified, no spills',flush=True)


if __name__=='__main__':main()
