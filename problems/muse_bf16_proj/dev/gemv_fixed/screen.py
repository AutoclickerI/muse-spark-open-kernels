"""Screen the complete MoE-hosted payload and its actual consumer glue."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
original=local_eval.load_module;records={}


def load(name,path):
    module=original(name,path)
    if name=='submission':
        def run(label,source,deadline,weights,state):
            import linecache
            filename='<h200_lab '+label+'>';linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
            ns=dict(module.__dict__);exec(compile(source,filename,'exec'),ns)
            start=time.monotonic();lines=ns['lab'](min(deadline,start+100.),weights,state)
            rows=ns['GF_ROWS'];records.update(rows=rows,lines=lines,elapsed_s=time.monotonic()-start)
            (HERE/'rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
            assert len([r for r in rows if r['kind']=='role'])==16,lines
            assert len([r for r in rows if r['kind']=='segment'])==4,lines
            assert all(len(r['trials'])==len(ns['GF_CONFIGS']) for r in rows if r['kind']=='role'),lines
            assert not any('ERROR' in line or 'INCOMPLETE' in line for line in lines),lines
            assert sum(len(line)+1 for line in lines)<=4450,lines
            records['admitted']=True
            return lines
        module._lab_run_one=run
    return module


local_eval.load_module=load
host=ROOT/'problems/muse_moe_w4a16/lab/lab_dense_gemv_fixed.py'
sys.argv=[sys.argv[0],str(host),'--tests-only','--no-baseline','--json',str(HERE/'host_local.json')]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        assert records.get('admitted'),'host passed but payload did not'
        report=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/dense_gemv_fixed.py').read_bytes()).hexdigest(),
            host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),roles=16,segments=4,configurations=14,
            host_tests=5,elapsed_s=records['elapsed_s'],log_characters=sum(len(x)+1 for x in records['lines']))
        (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('GF_HOST_ADMISSION',json.dumps(report),flush=True)
    raise
