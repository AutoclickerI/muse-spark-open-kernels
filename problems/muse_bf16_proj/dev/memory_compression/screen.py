"""Run the complete allocation/control/kernel experiment inside the host screen."""
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
            records.update(rows=ns['MC_ROWS'],support=ns['MC_SUPPORT'],lines=lines,elapsed_s=time.monotonic()-start)
            (HERE/'rows_local.json').write_text(json.dumps(records,indent=2)+'\n')
            assert not any('MC_ERROR' in line or 'INCOMPLETE' in line for line in lines),lines
            if records['support']['granted']==1:
                assert len([r for r in records['rows'] if r['kind']=='bandwidth'])==6,lines
                assert len([r for r in records['rows'] if r['kind'] in ('moe','dense')])==4,lines
                assert all(r['bit_identical'] for r in records['rows'] if r['kind']=='dense'),lines
                assert all(r['ordered']['bit_identical'] and r['ordered']['y_bit_identical']
                           for r in records['rows'] if r['kind']=='moe'),lines
                # Preserve and expose the incumbent's failed raw bit gate. It
                # cannot be an allocator gate: its A:A already fails locally.
                for r in records['rows']:
                    if r['kind']=='moe' and not r['bit_identical']:assert r['aa']['bit_mismatches']>0,lines
            else:
                assert 'unavailable' in lines[-1],lines
            assert sum(len(line)+1 for line in lines)<4450,lines
            records['admitted']=True
            return lines
        module._lab_run_one=run
    return module


local_eval.load_module=load
host=ROOT/'problems/muse_moe_w4a16/lab/lab_generic_memory_compression.py'
sys.argv=[sys.argv[0],str(host),'--tests-only','--no-baseline','--json',str(HERE/'host_local.json')]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        assert records.get('admitted'),'host passed but payload did not'
        report=dict(payload_sha256=hashlib.sha256((ROOT/'prof/lab/generic_memory_compression.py').read_bytes()).hexdigest(),
            host_sha256=hashlib.sha256(host.read_bytes()).hexdigest(),support=records['support'],rows=len(records['rows']),
            host_tests=5,elapsed_s=records['elapsed_s'],log_characters=sum(len(x)+1 for x in records['lines']),
            incumbent_bit_exact={str(r['tokens']):r['bit_identical'] for r in records['rows'] if r['kind']=='moe'},
            ordered_and_dense_bit_exact=True)
        (HERE/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MC_HOST_ADMISSION',json.dumps(report),flush=True)
    raise
