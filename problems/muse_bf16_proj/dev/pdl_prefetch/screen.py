"""Run the complete lab with forced Hopper attention Python glue on the local GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev')]
import local_eval
import verify_best

parser=argparse.ArgumentParser();parser.add_argument('--share-s',type=float,default=160.)
parser.add_argument('--host',default='muse_moe_w4a16')
args=parser.parse_args();original=local_eval.load_module;records={}
def load(name,path):
    module=original(name,path)
    if name=='submission':
        def run(label,source,deadline,weights,state):
            import linecache
            fname='<h200_lab '+label+'>';linecache.cache[fname]=(len(source),None,source.splitlines(True),fname)
            ns=dict(module.__dict__);exec(compile(source,fname,'exec'),ns)
            calls={}
            def install(m):
                calls['native_attention']=verify_best.install(m,False)
                original_fa3=m._attention_fa3_public
                def fa3(*a,**kw):
                    calls['fa3']=calls.get('fa3',0)+1
                    return original_fa3(*a,**kw)
                m._attention_fa3_public=fa3
            ns['PP_LOCAL_INSTALL']=install
            started=time.monotonic();lines=ns['lab'](min(deadline,started+args.share_s),weights,state)
            rows=ns['PP_ROWS'];records.update(rows=rows,lines=lines,elapsed_s=time.monotonic()-started)
            (HERE/'rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
            assert len([r for r in rows if r['kind']=='layer'])==8,lines
            assert len([r for r in rows if r['kind']=='stage'])==16,lines
            assert all(any(t['variant']=='W' for t in r['trials']) for r in rows if r['kind']=='stage'),lines
            assert 'synthetic=' in lines[1] and 'k/v_cache:E4M3' in lines[1],lines
            assert sum(len(l)+1 for l in lines)<=4450,lines
            assert not any('ERROR' in line or 'INCOMPLETE' in line for line in lines),lines
            assert all(not r['rejections'] for r in rows if r['kind']=='layer'),lines
            records['admitted']=True
            assert calls['native_attention'].get('attn_main',0)>0,calls
            assert calls.get('fa3',0)>0,calls
            records['forced_hopper_calls']=calls
            return lines
        module._lab_run_one=run
    return module
local_eval.load_module=load
host=ROOT/'problems'/args.host/'lab/lab_dense_pdl_prefetch.py'
sys.argv=[sys.argv[0],str(host),'--tests-only','--no-baseline','--json',str(HERE/'host_local.json')]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        assert records.get('admitted'),'host passed but lab did not'
        report=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/dense_pdl_prefetch.py').read_bytes()).hexdigest(),
            host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),host=args.host,layers=8,stages=16,host_tests=5,
            elapsed_s=records['elapsed_s'],share_s=args.share_s,forced_hopper=True,
            forced_hopper_calls=records['forced_hopper_calls'],
            log_characters=sum(len(l)+1 for l in records['lines']),provenance=records['lines'][1])
        (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('PDL_LOCAL_ADMISSION',json.dumps(report),flush=True)
    raise
