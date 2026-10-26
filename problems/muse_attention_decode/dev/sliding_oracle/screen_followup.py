"""Exact whole-layer packet with forced Hopper glue and explicit local cluster stand-ins."""
import hashlib,importlib.util,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch
import build_followup,source,structural
body=build_followup.build();fn='<p5-followup-local>';linecache.cache[fn]=(len(body),None,body.splitlines(True),fn);ns={};exec(compile(body,fn,'exec'),ns)
spec=importlib.util.spec_from_file_location('p5_so_force',R/'problems/muse_decode_layer/dev/best7/force.py');force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
spec=importlib.util.spec_from_file_location('p5_so_local_host',H.parent/'window_rebuild/local_host.py');local_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(local_host)
original=ns['follow_host']
def host():
    m=original();force.install(m,False,False);local_host.install(m);return m
ns['follow_host']=host;ns['LOCAL']=True
ns['SOURCES']={}
for f in ('d','v'):
    ns['SOURCES'][f+'OA']=source.original(f,True);ns['SOURCES'][f+'EA']=structural.source(f,True)
    ns['SOURCES'][f+'CLOCK']=structural.context_clock(f,False,True);ns['SOURCES'][f+'ECLOCK']=structural.context_clock(f,True,True)
ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+2400,None)
assert len(ns['RESULTS'])==4 and all(all(v==2 for v in r['TASK'].values()) for r in ns['RESULTS'])
assert all(all(c['TASK'] for c in r['clocks']) for r in ns['RESULTS'])
r=dict(payload_sha256=hashlib.sha256((H/'out/sliding_exp_pair.py').read_bytes()).hexdigest(),layer_TASK=40,producer_attention_TASK=8,
    archive_chars=len(next(x for x in lines if x.startswith('SLIDING_EXP_XZ85 '))),standins=['All S15 Hopper glue forced','decode C1 DSM every key','verify real two WGs/global last-WG merge','unrelated full N128 replaced by frozen S14 N64 arithmetic'],
    clock_glue=[dict(B=x['B'],L=x['L'],clocks=[{k:c[k] for k in ('candidate','TASK','gate_split','projection_role','partial_splits','partial_dtype','partial_stride')} for c in x['clocks']]) for x in ns['RESULTS']])
(H/'followup_admission.json').write_text(json.dumps(r,indent=2)+'\n');print('PASS',r)
