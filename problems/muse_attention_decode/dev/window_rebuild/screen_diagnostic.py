"""Screen the diagnostic's gates/captures/archive with an explicit portable FA3 stand-in."""
import ast,hashlib,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'tools'))
import torch
from build_diagnostic import build
body=build();fn='<p5-diagnostic-local>';linecache.cache[fn]=(len(body),None,body.splitlines(True),fn)
ns={};exec(compile(body,fn,'exec'),ns)
ns['LOCAL_STANDIN']=True;ns['CASES']=((1,2048),(16,2048),(32,2048));ns['SEEDS']=ns['SEEDS'][:1];ns['ROUNDS']=1
flush=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:flush.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+1500,None)
assert len(ns['RESULTS'])==3 and all(v==1 for r in ns['RESULTS'] for v in r['task'].values())
result=dict(payload_sha256=hashlib.sha256((H/'out/diagnostic.py').read_bytes()).hexdigest(),
            cases=3,arms=6,seeds=1,TASK_gates_passed=18,standin='FA3 call replaced by S15 portable attention math only; native split behavior and cost unmeasured locally',archive_chars=len(lines[1]))
(H/'diagnostic_admission.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
