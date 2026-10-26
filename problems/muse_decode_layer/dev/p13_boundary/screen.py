"""Exercise the exact wrapper, with S19's explicit local Hopper stand-ins."""
import builtins
import hashlib
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import p13_build
import h200_lab
from codec import unpack_samples
from lab_packet import read_packets
from local_eval import wrapper_smoke


def main():
    pair='--pair' in sys.argv
    if pair:
        import build_pair
        build_pair.main()
    else:p13_build.main()
    stem='p13_pair' if pair else 'p13_oracle'
    payload=(H/'out'/(stem+'.py')).read_text()
    sha=hashlib.sha256(payload.encode()).hexdigest()
    h200_lab.LAB_ONLY_LIMIT=14000
    wrapper,_=h200_lab.build([(stem,sha[:16],payload)],'muse_prefill_layer',phase='import',import_budget=820,lab_only=True)
    (H/'out'/(stem+'_wrapper.py')).write_text(wrapper)
    sp=importlib.util.spec_from_file_location('p13_screen',H.parent/'s19/screen.py')
    scr=importlib.util.module_from_spec(sp);sp.loader.exec_module(scr)
    def install(host):
        scr.install(host,True)
        # Unwrap local observer closures before cloning functions into independent
        # namespaces. Their captured _fn otherwise retains host RT_ARM globals.
        for name,fn in list(vars(host).items()):
            kw=getattr(fn,'__kwdefaults__',None) or {}
            if '_fn' in kw:setattr(host,name,kw['_fn'])
    emitted=[];archive=[]
    m=types.ModuleType('p13_wrapper');m.__file__=str(H/'out'/(stem+'_wrapper.py'))
    def emit(*args,**kwargs):
        line=kwargs.get('sep',' ').join(map(str,args))+kwargs.get('end','\n');emitted.append(line)
        if not line.startswith('LAB_PACKET '):builtins.print(*args,**kwargs)
    m.print=emit
    def inject():
        m.LOCAL_INSTALL=install
        m.LOCAL_ARCHIVE=lambda packet:archive.append(json.loads(json.dumps(packet)))
        original=m._lab_run_one
        def run(name,src,deadline,weights,state):
            if pair:src+='\nWEIGHT_SEED=130001003\nROUTER_SEED=130001009\nSEEDS=tuple(160001021+104729*i for i in range(8))\n'
            return original(name,src,time.monotonic()+3600,weights,state)
        m._lab_run_one=run
    m.P13_INJECT=inject
    marker='\n_lab_import_start = '
    assert wrapper.count(marker)==1
    print(wrapper_smoke('lab-only'),flush=True)
    exec(compile(wrapper.replace(marker,'\nP13_INJECT()'+marker,1),m.__file__,'exec'),m.__dict__)
    try:m.prepare(None);raise AssertionError('lab-only prepare returned')
    except RuntimeError as exc:assert 'lab-only host' in str(exc),exc
    records=read_packets((''.join(emitted)+'P'*2000)[-16000:])
    assert len(records)==1
    assert records[0]['packet'].get('complete'),records[0]['packet']
    packet=unpack_samples(records[0]['packet'])
    assert packet['complete'] and not packet.get('transport_errors'),packet
    assert records[0]['source_sha256']==sha[:16]
    for decoded,original in zip(packet['paired'],archive[0]['paired']):
        assert {k:v for k,v in decoded.items() if k not in ('round_index','telemetry_index')}==original
    assert all(v['pass_count']==96 and v['fail_count']==0 for v in packet['checks'].values())
    if pair:assert all(packet['nearby_checks'][name]==18 for name in ('INC','A','SMALL','LARGE'))
    wire=max(len(line.encode()) for line in emitted if line.startswith('LAB_PACKET '))
    result=dict(payload_sha256=sha,wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                checks=packet['checks'],wire_bytes=wire,roundtrip_exact=True,tail_with_2000_suffix_pass=True,
                protocol=packet['measurement_protocol'],scope='S19 forced Hopper glue/attention stand-ins on sm120; native timings unmeasured')
    if pair:result['local_fixture']='Oracle-screen weights130001003/router130001009, busiest-shard soft inputs160001021+104729*i; native confirmation fixed disjoint170001021 cohort and150001003/150001009 weights untouched. An earlier native-fixture local rehearsal failed INC v1F routing margin; unchanged S19 stand-in is not exact native attention.'
    (H/('pair_screen_validation.json' if pair else 'screen_validation.json')).write_text(json.dumps(result,indent=2)+'\n')
    (H/'out'/('pair_local_packet.json' if pair else 'local_packet.json')).write_text(json.dumps(packet)+'\n')
    print('PASS',result,flush=True)


if __name__=='__main__':main()
