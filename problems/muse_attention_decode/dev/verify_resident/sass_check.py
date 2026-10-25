"""Compile every native topology and verify early staging plus un-serialized WGMMA scheduling."""
import hashlib,json,re,subprocess
from pathlib import Path
import resgen
from build_payload import CONFIGS
H=Path(__file__).resolve().parent;rows=[]
for name,(cs,pipe) in CONFIGS.items():
    src=resgen.source(cs,pipe);p=H/'out'/(name+'.cu');p.write_text(src)
    r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True)
    log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log);assert r.returncode==0,(name,log[-2500:]);assert not re.search(r'C75\d+',log),(name,log[-2000:])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
    resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
    kernels=[]
    for block in sass.split('Function : ')[1:]:
        ins=[(int(m[1],16),m[2]) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\*',block)]
        waits=[pc for pc,op in ins if 'ACQBULK' in op]
        if not waits:continue
        first=min(waits);copies=[(hex(pc),op) for pc,op in ins if pc<first and 'LDGSTS' in op];trans=[(hex(pc),op) for pc,op in ins if pc<first and 'LDSM' in op]
        assert len(copies)==2*(2048//cs)//32,(name,len(copies));assert trans,(name,'no prewait V transpose')
        kernels.append(dict(symbol=block.splitlines()[0],waits=[hex(pc) for pc in waits],history_copy=copies,V_transpose=trans,
                            wait1=sum('WARPGROUP.DEPBAR' in op and '0x1' in op for _,op in ins)))
    assert len(kernels)==1,(name,len(kernels))
    if pipe and cs!=16:assert kernels[0]['wait1']>0,(name,'no WGMMA overlap wait')
    rows.append(dict(name=name,cs=cs,cap=2048//cs,pipeline=pipe,cuda_sha256=hashlib.sha256(src.encode()).hexdigest(),resources=resource,kernels=kernels,
                     dynamic_shared=4096+2*(2048//cs)*64+5120+8*32*8,no_C75=True))
    print('PASS',name,'history ops',len(kernels[0]['history_copy']),'wait1',kernels[0]['wait1'],flush=True)
(H/'out/native_admission.json').write_text(json.dumps(rows,indent=2)+'\n')
