"""Audit actual SM90a code, including N64 and asynchronous register-A lifetime."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'w13_issue'))
from count import parse


def lifetime(ins,name):
    loops=[]
    for pc,op,arg,pred in ins:
        m=re.search(r'0x([0-9a-f]+)',arg)
        if op=='BRA' and m and int(m[1],16)<pc:
            lo=int(m[1],16);body=[i for i in ins if lo<=i[0]<=pc]
            if sum(i[1].startswith('HGMMA.') for i in body)==8:loops.append((pc-lo,body))
    _,body=min(loops);pending=[];group=set();hazards=[]
    for k in range(32):
        for pc,op,arg,pred in body:
            if op.startswith('HGMMA.'):
                m=re.match(r'R\d+, R(\d+),',arg);assert m,(op,arg);r=int(m[1]);group.update(range(r,r+4))
                if 'gsb0' in arg:pending.append(group);group=set()
            elif op.startswith('WARPGROUP.DEPBAR'):
                n=int(re.search(r'0x([0-9a-f]+)$',arg)[1],16);pending=pending[-n:] if n else []
            elif pred!='@!PT':
                m=re.match(r'R(\d+)',arg)
                if m:
                    r=int(m[1]);width=4 if '.128' in op else 2 if '.64' in op else 1;dest=set(range(r,r+width));live=set().union(group,*pending)
                    if dest&live:hazards.append((k,hex(pc),op,arg,sorted(dest&live)))
    assert not hazards,(name,hazards[:3]);return len(body)


def main():
    report={}
    for key in ('c1','nt','opt','meta'):
        src=H/(key+'.cu');cube=src.with_suffix('.cubin')
        run=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(src),'-o',str(cube)],capture_output=True,text=True)
        src.with_suffix('.ptxas.log').write_text(run.stdout+run.stderr);run.check_returncode();assert 'C7513' not in run.stderr
        sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],capture_output=True,text=True,check=True).stdout;src.with_suffix('.sass').write_text(sass)
        regs={};name=None
        for line in run.stderr.splitlines():
            if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
            m=re.search(r'Used (\d+) registers',line)
            if m:regs[name]=int(m[1])
        rows=[]
        for part in sass.split('Function : ')[1:]:
            name=part.splitlines()[0].strip();ins=parse(part);local=sum(i[1].startswith(('LDL','STL')) for i in ins);assert not local,(name,local)
            mma=[i for i in ins if i[1].startswith('HGMMA.')];count=lifetime(ins,name) if mma else None
            waits=[i[2] for i in ins if i[1].startswith('WARPGROUP.DEPBAR')]
            if mma:assert len(waits)==3 and waits[-1].endswith('0x0'),(name,waits)
            if 'fusedILi64E' in name:assert all('64x64x16' in i[1] for i in mma)
            rows.append(dict(name=name,registers=regs[name],local_memory=local,waits=waits,loop_instructions=count,per16_weights=count/4 if count else None,live_A_overwrites=0 if mma else None))
        report[key]=dict(sha256=hashlib.sha256(src.read_bytes()).hexdigest(),kernels=rows)
    addr=0
    for n in (32,64):
        for shared in (0,4096,5120,8192,9216):
            base=(shared+1023)//1024*1024+18432
            for st in range(4):
                for chunk in range(2):
                    b=base+(st*n*128+chunk*n*64)*2
                    for row in range(n):
                        for k in range(64):
                            logical=b+row*128+k*2;reader=logical^(((logical>>7)&7)<<4);writer=b+(row*64+(k^((row&7)*8)))*2
                            assert reader==writer and b<=reader<b+n*128;addr+=1
    out=dict(entries=sum(len(r['kernels']) for r in report.values()),B128_addresses=addr,sources=report)
    (H/'native.json').write_text(json.dumps(out,indent=2)+'\n');print('MIX_NATIVE_PASS',out['entries'],addr,flush=True)


if __name__=='__main__':main()
