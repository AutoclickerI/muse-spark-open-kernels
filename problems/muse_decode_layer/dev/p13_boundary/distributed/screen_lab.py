"""Rehearse the exact complete lab transport on local explicit Hopper stand-ins."""
import argparse
import builtins
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import h200_lab
from codec import unpack_samples
from lab_packet import read_packets

def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=('boundary','route'));a=ap.parse_args()
    payload=(H/'out'/('p13_'+a.kind+'.py')).read_text();sha=hashlib.sha256(payload.encode()).hexdigest()
    h200_lab.LAB_ONLY_LIMIT=14000
    wrapper,_=h200_lab.build([('p13_'+a.kind,sha[:16],payload)],'muse_prefill_layer',phase='import',import_budget=820,lab_only=True)
    (H/'out'/(a.kind+'_wrapper.py')).write_text(wrapper)
    series=json.loads((H/(a.kind+'_lab_manifest.json')).read_text())['base_series'].lower()
    sp=importlib.util.spec_from_file_location('p13_production_screen',H.parents[1]/series/'screen.py');scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
    def install(host):
        scr.install(host,True)
        for name,fn in list(vars(host).items()):
            kw=getattr(fn,'__kwdefaults__',None) or {}
            if '_fn' in kw:setattr(host,name,kw['_fn'])
    emitted=[];archive=[];m=types.ModuleType('p13_'+a.kind);m.__file__=str(H/'out'/(a.kind+'_wrapper.py'))
    def emit(*args,**kwargs):
        line=kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n');emitted.append(line)
        if not line.startswith('LAB_PACKET '):builtins.print(*args,**kwargs)
    m.print=emit
    def inject():
        m.LOCAL_INSTALL=install
        def local_namespace(v):
            if a.kind=='boundary':v['P13B_LOCAL']=True
        m.LOCAL_NAMESPACE=local_namespace
        m.LOCAL_ARCHIVE=lambda packet:archive.append(json.loads(json.dumps(packet)))
        original=m._lab_run_one
        m._lab_run_one=lambda name,src,deadline,weights,state:original(name,src,time.monotonic()+3600,weights,state)
    m.P13_INSTALL=inject;marker='\n_lab_import_start = ';assert wrapper.count(marker)==1
    exec(compile(wrapper.replace(marker,'\nP13_INSTALL()'+marker,1),m.__file__,'exec'),m.__dict__)
    try:m.prepare(None);raise AssertionError('lab-only prepare returned')
    except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
    records=read_packets((''.join(emitted)+'S'*2000)[-16000:]);assert len(records)==1
    p=records[0]['packet'];assert p.get('complete') and not p.get('transport_errors'),p
    assert records[0]['source_sha256']==sha[:16];p=unpack_samples(p)
    for decoded,raw in zip(p['paired'],archive[0]['paired']):assert {k:v for k,v in decoded.items() if k not in ('round_index','telemetry_index')}==raw
    assert not p['failures'],p['failures']
    assert all(c==dict(pass_count=96,fail_count=0) for c in p['checks'].values())
    assert all(p['nearby_checks'][n]==18 for n in ('INC','A','B'))
    result=dict(payload_sha256=sha,wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),checks=p['checks'],nearby=p['nearby_checks'],
                wire_bytes=max(len(s.encode()) for s in emitted if s.startswith('LAB_PACKET ')),lossless_roundtrip=True,tail2000=True,
                scope='Exact '+series.upper()+' lab, full held-out fixture cohort; local Hopper attention/cluster stand-ins, native times unmeasured')
    (H/(a.kind+'_screen.json')).write_text(json.dumps(result,indent=2)+'\n');(H/'out'/(a.kind+'_local_packet.json')).write_text(json.dumps(p)+'\n')
    print('PASS',result,flush=True)

if __name__=='__main__':main()
