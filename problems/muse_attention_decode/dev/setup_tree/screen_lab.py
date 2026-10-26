"""Force exact S19 Hopper glue and C1 DSM substitutes in the packed lab."""
import hashlib,importlib.util,json,linecache,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer')]
import torch,pack_setup as pack,kernels_new as k
hoist='--hoist' in sys.argv
pack.build(hoist);body=(H/('out/hoist.py' if hoist else 'out/setup_tree.py')).read_text();payload_sha256=hashlib.sha256(body.encode()).hexdigest();fn='<p5-setup-tree-local>';linecache.cache[fn]=(len(body),None,body.splitlines(True),fn);ns={};exec(compile(body,fn,'exec'),ns)
native_sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in ns['SOURCES'].items()}
sp=importlib.util.spec_from_file_location('p5_setup_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
original=ns['setup_host']
def host():
    m=original();scr.install(m,True);return m
ns['setup_host']=host;ns['LOCAL']=True
ns['SOURCES']={f+a+'A':k.source(f,a,True) for f in ('d','v') for a in (('A','Q') if f=='d' else ('A','Q','T'))};ns['SOURCES'].update({f+'M':k.oracle(f,True) for f in ('d','v')})
if hoist:ns['SOURCES']={f+a+'A':k.source(f,a,True) for f in ('d','v') for a in ('A','H')}
ns['SEEDS']=ns['SEEDS'][:2];ns['ROUNDS']=2
buf=torch.empty(1<<30,device='cuda',dtype=torch.uint8);ns['lab_cold_l2']=lambda:buf.view(torch.int64).sum()
out=ns['lab'](time.monotonic()+3600,None);packet=out['packet'];assert packet['complete']
(H/('hoist_payload_admission.json' if hoist else 'payload_admission.json')).write_text(json.dumps(dict(payload_sha256=payload_sha256,checks=packet['checks'],standins='S19 forced Hopper glue; decode C1; verify one CTA/two WGs; all-key math and native coordinate arithmetic execute',native_sources=native_sources,local_sources=packet['identities']['sources']),indent=2)+'\n');print('PASS',packet['checks'])
