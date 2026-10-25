"""Audit the three diagnostic cubins after cuobjdump --dump-sass.

Usage: python3 roof_sass.py /tmp
Input names: p3b_roof{0,1,2}.sass. Counts are whole compiled bodies, not dynamic
retired counts. The matching LDGSTS/LDG and MMA counts test ablation parity.
"""
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def main():
    root=Path(sys.argv[1]);result={}
    for i,mode in enumerate(('unchanged','trivial','loads')):
        src=root/f'p3b_roof{i}.sass';records={}
        for block in src.read_text().split('Function : ')[1:]:
            name=block.splitlines()[0]
            if 'mwbr4gemmI' in name:
                key='heavy_W13' if 'ELb1E' in name else 'heavy_W2'
            elif 'pmz5phaseILi1E' in name and 'ELi16E' in name:
                key='NT1_W13' if 'phaseILi1ELb1E' in name else 'NT1_W2'
            else:continue
            ops=[]
            for line in block.splitlines():
                m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?[UP][0-9]+\s+)?([A-Z][A-Z0-9_.]*)',line)
                if m:ops.append(m[1])
            counts=collections.Counter(op.split('.')[0] for op in ops)
            records[key]=dict(instructions=len(ops),ops=dict(sorted(counts.items())))
        result[mode]=dict(sass_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),kernels=records)
    h=Path(__file__).parent
    (h/'roof_sass.json').write_text(json.dumps(result,indent=2)+'\n')
    for key in ('heavy_W13','heavy_W2','NT1_W13','NT1_W2'):
        print(key)
        for mode in result:
            r=result[mode]['kernels'][key];c=r['ops']
            print(mode,r['instructions'],{k:c.get(k,0) for k in ('LDG','LDGSTS','LDS','HGMMA','HMMA','HMUL2','PRMT','SHF')})


if __name__=='__main__':main()
