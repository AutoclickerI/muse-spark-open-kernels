"""Normal/forced local_eval audit of the exact wrapped decode lab, with the Hopper-only ring's launch stand-in."""
import argparse,importlib.util,linecache,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
original=local_eval.load_module

def loader(name,path):
    m=original(name,path)
    if name!='submission':return m
    def run(name,src,deadline,weights,state):
        fn='<local_p5_full128_wrapped_lab>'
        linecache.cache[fn]=(len(src),None,src.splitlines(True),fn);ns=dict(m.__dict__)
        exec(compile(src,fn,'exec'),ns)
        ns['PF_CLOCKS']=False
        ns['PF_MOD'][False]=m._vq8_module()
        ns['PF_SEEDS']=ns['PF_SEEDS'][:2];ns['PF_ROUNDS']=2
        if 'CS_SEEDS' in ns:ns['CS_SEEDS']=ns['CS_SEEDS'][:2];ns['CS_ROUNDS']=2
        lines=ns['lab'](deadline,weights,state)
        assert len(ns['PF_RESULTS'])==2
        if 'CS_RESULTS' in ns:
            assert len(ns['CS_RESULTS'])==3
            assert all(len(p['nodes'])==1 for r in ns['CS_RESULTS'] for n,p in r['profiles'].items() if n in ('C64','C128','W4','W8')), 'last-CTA capture must contain exactly one attention kernel'
        return ['LOCAL N128 persistent ring uses explicit S11 launch stand-in; exact N128 math independently passes check_tile.py.']+lines
    m._lab_run_one=run
    return m
local_eval.load_module=loader
spec=importlib.util.spec_from_file_location('p5_audit',ROOT/'problems/muse_decode_layer/dev/s9/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
audit.main()
