"""Full exact wrapper + all off arms, with local-only Hopper launch stand-ins."""
import importlib.util
import json
import sys
import time
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval
import local_standins
forced='--forced' in sys.argv
original_loader=local_eval.load_module
records=[]


def install(module):
    if forced:
        spec=importlib.util.spec_from_file_location('s15_best7_force',H.parent/'best7/force.py')
        force=importlib.util.module_from_spec(spec);spec.loader.exec_module(force)
        force.install(module,False,False)
    return local_standins.install(module)


def loader(name,path):
    m=original_loader(name,path)
    if name=='submission':
        install(m)
        m.S15_LOCAL_FORCE=install
        run=m._lab_run_one
        def checked(name,source,deadline,weights,state):
            # Keep all 16 seeds, all timing rounds and the full read flush locally.
            # The slower sm120 screen gets more wall time; the shipped deadline is unchanged.
            lines=run(name,source,time.monotonic()+1800,weights,state)
            records.extend(lines)
            return lines
        m._lab_run_one=checked
    return m


local_eval.load_module=loader
target=R/'problems/muse_decode_layer/lab/lab_s15ab_payload.py'
out=H/('lab_forced_local.json' if forced else 'lab_normal_local.json')
sys.argv=['local_eval.py',str(target),'--tests-only','--json',str(out)]
try:local_eval.main()
except SystemExit as exc:
    if exc.code==0:
        rows=[json.loads(line.removeprefix('RESULT ')) for line in records if line.startswith('RESULT ')]
        assert len(rows)==1,records[-3:]
        raw=rows[0]
        (H/('lab_forced_validation.json' if forced else 'lab_normal_validation.json')).write_text(json.dumps(raw,indent=2)+'\n')
        assert raw['complete'] and not raw['failures'] and not raw['errors'],raw
    raise
