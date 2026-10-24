"""Compile every pipeline specialization for sm90a; reject spills or serialized WGMMA."""
import json
from pathlib import Path
import re
import subprocess
from pipeline import PIPE_CONFIGS

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics'


def main():
    src=(HERE/'pipeline.cu').read_text()
    src='#include <cuda_bf16.h>\n#include <stdint.h>\n'+src[src.index('namespace mwpipe'):src.index('constexpr int CONFIGS')]+'\n}\n'
    for nt,kt,st,sk,overlap in PIPE_CONFIGS:
        src+=f'template __global__ void mwpipe::grouped<{nt},{kt},{st},{sk},{str(overlap).lower()}>(const int*,const unsigned char*,const float*,const __nv_bfloat16*,const int*,const int*,const int*,const int*,const int*,float*,int,int,int,int);\n'
    path=OUT/'pipeline.cu';path.write_text(src)
    p=subprocess.run(['nvcc','--cubin','-O3','-lineinfo','-arch=sm_90a','-std=c++20','-Xptxas=-v',
                      str(path),'-o',str(path.with_suffix('.cubin'))],capture_output=True,text=True)
    (OUT/'pipeline.ptxas.log').write_text(p.stdout+p.stderr);p.check_returncode()
    assert 'C7513' not in p.stderr,'Compiler serialized WGMMA; inspect pipeline.ptxas.log'
    s=subprocess.run(['cuobjdump','--dump-sass',str(path.with_suffix('.cubin'))],capture_output=True,text=True,check=True).stdout
    path.with_suffix('.sass').write_text(s);out=[];regs={};current=None
    for line in p.stderr.splitlines():
        if 'Function properties for ' in line:current=line.split('Function properties for ',1)[1].strip()
        match=re.search(r'Used (\d+) registers',line)
        if match:regs[current]=int(match[1])
    for part in s.split('Function : ')[1:]:
        name=part.splitlines()[0].strip()
        if 'grouped' not in name:continue
        cfg=re.search(r'groupedILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELb([01])E',name)
        assert cfg,name
        nt,kt,st,sk,overlap=map(int,cfg.groups())
        ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Za-z][A-Za-z0-9_.]*)\s*(.*?)\s*;',part)
        waits=[arg for op,arg in ops if op.startswith('WARPGROUP.DEPBAR')]
        row={'config':[nt,kt,st,sk,bool(overlap)],'registers':regs[name],'local_load_store':sum(op.startswith(('LDL','STL')) for op,_ in ops),
             'static_bf16_mma':sum(op.startswith('HGMMA.') and '.BF16' in op for op,_ in ops),
             'static_cp_async':sum(op.startswith('LDGSTS') for op,_ in ops),'waits':waits,
             'static_CTA_barrier':sum(op.startswith('BAR.') for op,_ in ops),
             'stage_packed_Q_SF_bytes':64*kt*9//16,'prologue_packed_Q_SF_bytes':(st-1)*64*kt*9//16}
        assert row['local_load_store']==0,row
        if overlap:assert len(waits)==3 and waits[0].endswith('0x1') and waits[1].endswith('0x1') and waits[2].endswith('0x0'),row
        out.append(row)
    assert len(out)==len(PIPE_CONFIGS),len(out)
    (OUT/'pipeline_sass_counts.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
