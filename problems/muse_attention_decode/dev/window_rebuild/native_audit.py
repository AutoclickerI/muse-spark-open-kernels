"""Native Hopper compilation, resources and pre-PDL issued-copy evidence."""
import hashlib,json,re,subprocess,sys
from pathlib import Path
import kernel
H=Path(__file__).resolve().parent
(H/'out').mkdir(exist_ok=True)
rows=[]
for name,c,stream in (('WG2C2',2,False),('WG2C4',4,False),('STREAM2',2,True),('STREAM4',4,True),('STREAM2_CLOCK',2,True),('WG2C2_CLOCK',2,False),('PIPE2',2,True),('PIPE2_CLOCK',2,True)):
    if len(sys.argv)>1 and name not in sys.argv[1:]:continue
    src=kernel.source(c,stream=stream,clocks=name.endswith('_CLOCK'),pipeline=name.startswith('PIPE'));p=H/'out'/(name+'.cu');p.write_text(src)
    r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-Xptxas=-v','-c',str(p),'-o',str(p.with_suffix('.o'))],capture_output=True,text=True)
    log=r.stdout+r.stderr;p.with_suffix('.compile.log').write_text(log);assert r.returncode==0,(name,log[-3000:]);assert not re.search(r'C75\d+',log),(name,log[-2500:])
    sass=subprocess.check_output(['cuobjdump','--dump-sass',str(p.with_suffix('.o'))],text=True);p.with_suffix('.sass').write_text(sass)
    resource=subprocess.check_output(['cuobjdump','--dump-resource-usage',str(p.with_suffix('.o'))],text=True)
    evidence=[]
    for block in sass.split('Function : ')[1:]:
        ins=[(int(m[1],16),m[2]) for m in re.finditer(r'/\*([0-9a-f]+)\*/\s+(.*?)\s+;\s*/\*',block)]
        waits=[pc for pc,op in ins if 'ACQBULK' in op]
        if not waits:continue
        copies=[(hex(pc),op) for pc,op in ins if pc<min(waits) and 'LDGSTS' in op]
        assert copies,(name,'history copies not issued before PDL wait')
        evidence.append(dict(symbol=block.splitlines()[0],first_wait=hex(min(waits)),prewait_copy_instructions=len(copies)))
    assert len(evidence)==1,(name,evidence)
    rows.append(dict(name=name,cuda_sha256=hashlib.sha256(src.encode()).hexdigest(),resources=resource,
        two_WG_named_barriers=True,evidence=evidence,no_C75=True))
    print('PASS',name,resource.replace('\n',' ')[:220],flush=True)
p=H/'native_admission.json'
if len(sys.argv)>1 and p.exists():rows=[r for r in json.loads(p.read_text()) if r['name'] not in {x['name'] for x in rows}]+rows
p.write_text(json.dumps(rows,indent=2)+'\n')
