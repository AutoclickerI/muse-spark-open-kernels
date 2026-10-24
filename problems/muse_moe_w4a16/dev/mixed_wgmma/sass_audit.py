"""Count steady K-loop SASS, including the skipped special-case path, from actual sm90a cubins."""
import collections
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent


def audit(path,k=128):
    ins=[]
    for line in path.read_text().splitlines():
        m=re.match(r'\s*/\*([0-9a-f]+)\*/\s+(?:(@!?P\d+)\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',line)
        if m:ins.append((int(m[1],16),m[3],m[4],m[2]))
    loops=[]
    for pc,op,args,pred in ins:
        m=re.match(r'0x([0-9a-f]+)',args)
        if op=='BRA' and m and int(m[1],16)<pc:
            lo=int(m[1],16);body=[i for i in ins if lo<=i[0]<=pc]
            if sum(i[1].startswith('HGMMA.') and '.BF16' in i[1] for i in body)==k//16:loops.append((pc-lo,lo,pc,body))
    _,lo,hi,body=min(loops)
    counts=collections.Counter(i[1] for i in body)
    conv=[i for i in body if i[1].startswith(('HMUL2','PRMT','FLO','LOP3','SHF','ISETP','BSSY','BSYNC','BRA'))]
    report={'file':str(path),'loop_pc':[hex(lo),hex(hi)],'static_loop_instructions':len(body),
            'K_step':k,'weights_per_thread_per_step':k//2,'packed_bytes_per_thread_per_step':k//4,
            'instructions_per_packed_byte':len(body)/(k//4),'conversion_related_static_instructions':len(conv),
            'local_load_store':sum(n for op,n in counts.items() if op.startswith(('LDL','STL'))),
            'opcodes':dict(sorted(counts.items())),
            'note':'Static instruction count includes both normal and subnormal branches; not dynamic executed instructions or measured cycles.'}
    return report


if __name__=='__main__':
    reports={name:audit(HERE/'diagnostics'/f'{name}.sass') for name in ('scalar','plane')}
    reports['vector32']=audit(HERE/'diagnostics/vector32.sass',64)
    (HERE/'diagnostics/sass_counts.json').write_text(json.dumps(reports,indent=2)+'\n')
    print(json.dumps(reports,indent=2))
