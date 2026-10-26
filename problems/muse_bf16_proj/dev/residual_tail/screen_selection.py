"""Official local_eval screen of the exact per-shape production composition."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time
import types

H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer'
sys.path[:0]=[str(R/'tools'),str(P)]
import local_eval

p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args()
points=json.loads((H/'point_nominations.json').read_text());selection=points['selected']
path=R/selection['candidate'];assert hashlib.sha256(path.read_bytes()).hexdigest()==selection['candidate_sha256']
ref=local_eval.load_module('reference',P/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
mod=local_eval.load_module('p7_rt_selection',path)
if a.forced:
    force=local_eval.load_module('p7_rt_force',P/'dev/best7/force.py');force.install(mod,False,False)
stand=local_eval.load_module('p7_rt_standins',P/'dev/s15/local_standins.py');stand.install(mod)
spec=importlib.util.spec_from_file_location('p7_d1_screen',H.parent/'d1_chain/screen.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q);local_eval.Timer.replay_us=q.replay_us
weights=ref.generate_weights(ref.WEIGHTS_SEED);started=time.monotonic();state=mod.prepare(weights)
problem=local_eval.Problem(ref,weights);side=types.SimpleNamespace(call=lambda data:mod.kernel_fn(data,state),prepare_s=0.)
checks=local_eval.run_tests(problem,side,local_eval.tests_of((P/'TASK.md').read_text()),210000)
assert len(checks)==13 and all(v['ok'] for v in checks)
benches=json.loads((R/'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks']
timer=local_eval.Timer();rows=[local_eval.time_shape(problem,[side],case,2,1,timer) for case in benches]
assert len(rows)==12 and all(v['ok'] for v in rows)
x=dict(forced=a.forced,selection=selection,tests=checks,shapes=rows,elapsed_s=time.monotonic()-started)
(H/'out'/('selection_forced.json' if a.forced else 'selection_normal.json')).write_text(json.dumps(x,indent=2)+'\n')
print('PASS selected composition',len(checks),'TASK tests',len(rows),'captured shapes',flush=True)
