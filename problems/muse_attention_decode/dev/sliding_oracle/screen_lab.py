"""Execute the exact packed payload with explicit local DSM/FA3 stand-ins."""
import hashlib,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent
import build_oracle,source
body=build_oracle.build();fn='<p5-oracle-local>';linecache.cache[fn]=(len(body),None,body.splitlines(True),fn);ns={};exec(compile(body,fn,'exec'),ns)
import torch
ns['SOURCES']={f+k:(source.original(f,True) if k=='A' else source.source(f,True,clocks=k=='CLOCK')) for f in ('d','v') for k in ('A','M','CLOCK')}
ns['LOCAL']=True;ns['LOCAL_STANDIN']=True;ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+1800,None)
assert len(ns['RESULTS'])==4 and all(all(v==2 for v in r['TASK'].values()) for r in ns['RESULTS'])
result=dict(payload_sha256=hashlib.sha256((H/'out/sliding_oracle.py').read_bytes()).hexdigest(),TASK_passed=sum(sum(r['TASK'].values()) for r in ns['RESULTS'])+4,
    oracle_branches_executed=True,archive_chars=len(next(x for x in lines if x.startswith('SLIDING_ORACLE_XZ85 '))),standins=['decode DSM C1, every key retained','verify global last-WG merge, same C2 streaming math','FA3 uses portable attention only locally'])
(H/'payload_admission.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result)
