"""Execute the exact batched payload locally with forced Hopper glue and two fresh seeds."""
import importlib.util,linecache,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer')]
import torch
spec=importlib.util.spec_from_file_location('force',ROOT/'problems/muse_decode_layer/dev/best7/force.py');force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
source=(HERE/'out/p5_early_history_board.py').read_text();fn='<p5-early-history-board-local>';linecache.cache[fn]=(len(source),None,source.splitlines(True),fn)
ns={'__name__':'p5_local_board','__file__':fn};exec(compile(source,fn,'exec'),ns)
payload=ns['_LAB_PAYLOADS'][0][2];pn='<p5-history-payload-local>';linecache.cache[pn]=(len(payload),None,payload.splitlines(True),pn)
g=dict(ns);exec(compile(payload,pn,'exec'),g)
original=g['eh_host']
def local_host(src):
    host=original(src);force.install(host,False,False)
    host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return host
g['eh_host']=local_host
g['EH_SEEDS']=g['EH_SEEDS'][:2];g['EH_ROUNDS']=2
lines=g['lab'](time.monotonic()+1200,None,None)
assert len(g['EH_RESULTS'])==1
assert all(len(p['nodes'])==1 for n,p in g['EH_RESULTS'][0]['profiles'].items() if n not in ('A','A2'))
print('\n'.join(x for x in lines if not x.startswith('EH_XZ85 ')))
import json
(HERE/'lab_local.json').write_text(json.dumps(g['EH_RESULTS'],indent=2)+'\n')
print('PASS exact payload, forced whole-layer glue, seven arms, two fresh seeds x2 cold rounds')
