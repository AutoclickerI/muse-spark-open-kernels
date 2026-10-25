"""Audit SASS/resources of the lean TMA, parallel-copy and direct-gather arms."""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys
H=Path(__file__).resolve().parent
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp')
result={}
for mode in ('lean','copy','direct','deep'):
    log=(root/f'p3b_cluster_{mode}_ptxas.log').read_text();rows=[]
    for b in log.split('Compiling entry function ')[1:]:
        z=re.search(r'oc6kernelILi(\d+)ELb(\d)ELi(\d+)',b.splitlines()[0])
        if not z:continue
        sm=re.search(r'(\d+) bytes smem',b)
        fr=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',b)
        nt,first,cs=map(int,z.groups());st=3 if mode=='deep' and nt<=64 else 2 if nt<=128 else 1
        rows.append(dict(maxnt=nt,first=first,cs=cs,registers=int(re.search(r'Used (\d+) registers',b)[1]),
                         static_smem=int(sm[1]) if sm else 0,
                         dynamic_smem=((st*4608+1023)//1024)*1024+st*nt*256,
                         stack_spill=list(map(int,fr.groups()))))
    s=(root/f'p3b_cluster_{mode}.sass').read_text();kernels={}
    for b in s.split('Function : ')[1:]:
        z=re.search(r'oc6kernelILi(\d+)ELb(\d)ELi(\d+)',b.splitlines()[0])
        if not z:continue
        nt=int(z[1]);st=3 if mode=='deep' and nt<=64 else 2 if nt<=128 else 1
        ins=[(int(a,16),v) for a,v in re.findall(r'/\*([0-9a-f]+)\*/\s+(.+?);',b)];loops=[]
        for pc,v in ins:
            br=re.search(r'BRA (0x[0-9a-f]+)',v)
            if br and int(br[1],16)<pc:
                body=[x for a,x in ins if int(br[1],16)<=a<=pc];hg=[x for x in body if 'HGMMA' in x]
                if len(hg)==8*st:
                    loops.append(dict(nt=int(re.search(r'HGMMA.64x(\d+)x',hg[0])[1]),
                                      KT128_instructions=len(body)/st,HGMMA=8,
                                      HMUL2=sum('HMUL2' in x for x in body)/st,
                                      LDGSTS=sum('LDGSTS' in x for x in body)/st,
                                      TMA=sum('UTMALDG' in x for x in body)/st,
                                      local=sum('LDL' in x or 'STL' in x for x in body)))
        loops=[x for x in loops if x['KT128_instructions']==min(y['KT128_instructions'] for y in loops if y['nt']==x['nt'])]
        kernels['/'.join(z.groups())]=dict(loops=loops,depbar=dict(collections.Counter(re.findall(r'WARPGROUP.DEPBAR.LE gsb0, (0x[0-9a-f]+)',b))))
    assert rows and all(not any(x['stack_spill']) for x in rows),(mode,'spills')
    assert 'C7512' not in log,(mode,'WGMMA serialization warning')
    result[mode]=dict(cuda_sha256=hashlib.sha256((H/f'cu_cluster_{mode}.cu').read_bytes()).hexdigest(),
                      sass_sha256=hashlib.sha256(s.encode()).hexdigest(),resources=rows,kernels=kernels)
    print(mode,[(x['maxnt'],x['first'],x['registers'],x['static_smem']) for x in rows if x['cs']==1])
    print('LOOPS',[(k,[x['KT128_instructions'] for x in v['loops']]) for k,v in kernels.items() if k.endswith('/1')])
(H/'pipeline_resources.json').write_text(json.dumps(result,indent=2)+'\n')
