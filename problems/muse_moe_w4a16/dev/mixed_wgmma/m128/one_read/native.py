"""Compile actual Hopper paths; audit SS descriptors and RS register lifetimes."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from check_async import sass_paths


def address_gate():
    writes=set()
    for t in range(128):
        w=t//32;l=t%32
        for atom in range(4):
            for j in range(4):
                r=w*16+l//4+(j&1)*8;k=atom*16+(l&3)*2+(j>>1)*8
                for half in (0,1):
                    p=r*64+((k^((r&7)*8))+half)
                    assert p not in writes
                    writes.add(p)
    assert writes==set(range(4096))
    checked=0
    for base in (0,1024,19456,35840,52224):
        assert base%1024==0
        for r in range(64):
            for atom in range(4):
                for k in range(16):
                    logical=base+r*128+(atom*16+k)*2
                    canonical=logical^(((logical>>7)&7)<<4)
                    writer=base+(r*64+((atom*16+k)^((r&7)*8)))*2
                    assert writer==canonical
                    checked+=1
    return dict(unique_A_half_values=len(writes),canonical_B128_addresses=checked)


def main():
    result={};address=address_gate()
    for prefix in ('or_rs','or_ss64','or_ss128','norig'):
        p=HERE/(prefix+'.cu');cube=p.with_suffix('.cubin')
        r=subprocess.run(['nvcc','--cubin','-O3','-std=c++20','-arch=sm_90a','-lineinfo','-Xptxas=-v',str(p),'-o',str(cube)],capture_output=True,text=True)
        p.with_suffix('.ptxas.log').write_text(r.stdout+r.stderr);r.check_returncode()
        assert 'C7513' not in r.stderr,r.stderr
        sass=subprocess.run(['cuobjdump','--dump-sass',str(cube)],capture_output=True,text=True,check=True).stdout
        p.with_suffix('.sass').write_text(sass);rows=[];info={};name=None
        for line in r.stderr.splitlines():
            if 'Function properties for ' in line:name=line.split('Function properties for ',1)[1].strip()
            m=re.search(r'Used (\d+) registers',line)
            if m:info[name]=int(m[1])
        for part in sass.split('Function : ')[1:]:
            name=part.splitlines()[0].strip()
            if 'fused' not in name:continue
            ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part)
            local=sum(op.startswith(('LDL','STL')) for op,arg in ops)
            target='ILi128ELi64ELi1ELi1' in name if prefix.startswith('or_ss') else True
            # Extra384-token configurations test sharing beyond N128. Record
            # their resources separately; no nearby-shape spill is hidden.
            if target:assert local==0,(name,local)
            mma=[(op,arg) for op,arg in ops if op.startswith(('HGMMA.','HMMA.'))]
            if prefix.startswith('or_ss'):
                # HGMMA SS packs the A/B descriptor pair into one gdesc uniform
                # register group. RS has an extra register-A operand between
                # the destination and gdesc; it must be absent on this path.
                assert mma and all(re.match(r'R\d+, gdesc\[UR\d+\], R\d+',arg) for op,arg in mma),(name,mma[:1])
                waits=[arg for op,arg in ops if op.startswith('WARPGROUP.DEPBAR')]
                assert waits and all(arg.endswith('0x0') for arg in waits),(name,waits)
            rows.append(dict(name=name,registers=info[name],local_memory_instructions=local,target_M128=target,mma_instructions=len(mma)))
        result[prefix]=dict(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),kernels=rows)
    paths=[p for k in sass_paths(HERE/'or_rs.sass') for p in k['paths']]
    assert len(paths)==10 and all(p['hazard_count']==0 for p in paths),paths
    (HERE/'native.json').write_text(json.dumps(dict(sources=result,address=address,RS_safe_K64_paths=len(paths)),indent=2)+'\n')
    print('ONE_READ_NATIVE_PASS',sum(len(x['kernels']) for x in result.values()),'GEMMs;',len(paths),'safe RS paths;',address)


if __name__=='__main__':main()
