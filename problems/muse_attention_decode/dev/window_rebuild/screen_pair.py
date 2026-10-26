"""Execute the exact pair with forced S15 glue and explicit local kernel/FA3 stand-ins."""
import hashlib,importlib.util,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch,kernel,local_host
from build_pair import build
pipeline='--pipeline' in sys.argv
body=build(pipeline);fn='<p5-window-pair-local>';linecache.cache[fn]=(len(body),None,body.splitlines(True),fn)
ns={};exec(compile(body,fn,'exec'),ns)
spec=importlib.util.spec_from_file_location('window_force',R/'problems/muse_decode_layer/dev/best7/force.py');force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
original=ns['window_host']
def host():
    m=original();force.install(m,False,False);local_host.install(m);return m
ns['window_host']=host
ns['wr_KERNELS']={n:kernel.source(ns['wr_TOPOLOGY'][n.removesuffix('_CLOCK')],local=True,stream=n.startswith(('STREAM','PIPE')),clocks=n.endswith('_CLOCK'),pipeline=n.startswith('PIPE')) for n in ns['wr_KERNELS']}
ns['fa3_eight']=lambda host,qkv,d,st:host._best7_attention(qkv,d,st,st['best7_plans'][(d['config']['batch'],8,2048)])
ns['LOCAL_STANDIN']=True;ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
lines=ns['lab'](time.monotonic()+1800,None)
assert len(ns['RESULTS'])==2 and all(v==2 for r in ns['RESULTS'] for v in r['TASK'].values())
arms=len(ns['RESULTS'][0]['TASK'])
result=dict(payload_sha256=hashlib.sha256((H/('out/pipeline_pair.py' if pipeline else 'out/pair.py')).read_bytes()).hexdigest(),shapes=2,arms=arms,seeds=2,TASK_passed=4*arms,
    standins=['all S15 Hopper glue forced','native two-WG DSM replaced by nonspinning global last-WG merge','residentC2 uses C4 capacity on sm120; streaming slice geometry unchanged','unrelated S15 full N128 uses frozen N64 math','FA3 split8 arm uses baseline portable attention; native API already checked by45ab5a9b'],
    archive_chars=len(next(x for x in lines if x.startswith('WINDOW_PAIR_XZ85 '))),clocks=[r['extra']['clocks'] for r in ns['RESULTS']])
(H/('pipeline_pair_admission.json' if pipeline else 'pair_admission.json')).write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['TASK_passed'],'TASK gates; archive chars',result['archive_chars'])
