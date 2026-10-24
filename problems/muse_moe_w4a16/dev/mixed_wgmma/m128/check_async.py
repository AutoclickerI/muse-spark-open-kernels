"""Audit pending RS-register lifetimes and descriptor addresses on native SASS.

This is an ordering/address check, not a simulation of unspecified Hopper values.
The portable arithmetic path consumes its input immediately and cannot model UB.
"""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent


def pending_hazards(banks, stages=64):
    pending=[];hazards=[]
    for stage in range(stages):
        bank=banks[stage%len(banks)]
        overlap=sorted(set(bank).intersection(r for p in pending for r in p))
        if overlap:hazards.append(dict(stage=stage,overwritten=overlap))
        pending.append(bank)
        # wgmma.commit_group; wait_group 1 may leave this latest group live.
        pending=pending[-1:]
    return hazards


def sass_paths(path):
    rows=[]
    for part in path.read_text().split('Function : ')[1:]:
        name=part.splitlines()[0].strip()
        if 'ILi128ELi64ELi4ELi1' not in name:continue
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part)
        paths=[];groups=[];fragment=[];width=None
        for op,arg in ops:
            if op.startswith('HGMMA.'):
                width=int(re.search(r'64x(\d+)x16',op)[1])
                a=int(re.match(r'R\d+, R(\d+),',arg)[1]);fragment+=list(range(a,a+4))
                if 'gsb0' in arg:groups.append(fragment);fragment=[]
            if op.startswith('WARPGROUP.DEPBAR') and arg.endswith('0x0'):
                if groups:paths.append(dict(width=width,banks=groups,hazard_count=len(pending_hazards(groups)),first_hazard=next(iter(pending_hazards(groups)),None)))
                groups=[]
        rows.append(dict(kernel=name,paths=paths))
    return rows


def address_checks():
    # Canonical CuTe Major-K B128 layout in byte units: ((8,n),k)
    # strides ((128,1024),2), followed by Swizzle<3,4,3> on the address.
    n=0
    for nt,kt,stages in ((128,64,4),(128,128,2),(64,128,4)):
        bs=(stages*(64*kt//2+64*kt//16)+1023)&~1023
        for static_bytes in (17408,3712,9216,2432):
            sm=(static_bytes+1023)&~1023
            for stage in range(stages):
                for chunk in range(kt//64):
                    base=sm+bs+(stage*nt*kt+chunk*nt*64)*2
                    assert base%1024==0
                    for r in range(nt):
                        for b in range(4):
                            for k in range(16):
                                logical=base+(r//8)*1024+(r%8)*128+(b*16+k)*2
                                reader=logical^(((logical>>7)&7)<<4)
                                writer=base+(r*64+((b*16+k)^((r&7)*8)))*2
                                assert reader==writer,(nt,kt,stage,r,b,k,reader,writer)
                                assert base<=reader<base+nt*64*2
                                n+=1
    return n


def main():
    before=sass_paths(HERE/'native_before.sass');after=sass_paths(HERE/'native.sass')
    assert len(before)==len(after)==4
    old=[p for k in before for p in k['paths']];new=[p for k in after for p in k['paths']]
    assert len(old)==len(new)==10
    assert all(len(p['banks'])==1 and p['hazard_count']==63 for p in old)
    assert all(len(p['banks'])==2 and p['hazard_count']==0 for p in new)
    addresses=address_checks()
    report=dict(before=before,after=after,address_pairs=addresses,
        ptx='https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group',
        conclusion='Old K64 reuses live A registers before the wait retiring their group. Fixed K64 uses disjoint banks; wait1 retires a bank before its reuse. This proves a PTX lifetime violation, not a numerical replay of undefined Hopper behavior.')
    (HERE/'async_lifetime.json').write_text(json.dumps(report,indent=2)+'\n')
    print('ASYNC_LIFETIME_PASS',len(old),'old unsafe paths;',len(new),'fixed safe paths;',addresses,'Hopper B128 addresses match writer')


if __name__=='__main__':main()
