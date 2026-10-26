"""Official local_eval gates for every arm, sharing only unchanged S15 prepare."""
import argparse,hashlib,importlib.util,json,pathlib,sys,time,types
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];P=R/'problems/muse_decode_layer';sys.path[:0]=[str(R/'tools'),str(P)]
import torch,local_eval
p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args()
ref=local_eval.load_module('reference',P/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
mod=local_eval.load_module('p7_mt_screen',H/'F4_trial.py')
if a.forced:
 force=local_eval.load_module('p7_mt_force',P/'dev/best7/force.py');force.install(mod,False,False)
stand=local_eval.load_module('p7_mt_standins',P/'dev/s15/local_standins.py');stand.install(mod)
# Use the project's existing bounded local graph timing implementation.
sp=importlib.util.spec_from_file_location('p7_d1_screen',H.parent/'d1_chain/screen.py');q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q);local_eval.Timer.replay_us=q.replay_us
weights=ref.generate_weights(ref.WEIGHTS_SEED);started=time.monotonic();state=mod.prepare(weights)
problem=local_eval.Problem(ref,weights);tests=local_eval.tests_of((P/'TASK.md').read_text());benches=json.loads((R/'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks'];timer=local_eval.Timer();result={}
for arm in mod.MT_ARMS:
 mod.MT_ARM=arm;side=types.SimpleNamespace(call=lambda data:mod.kernel_fn(data,state),prepare_s=0.)
 checks=local_eval.run_tests(problem,side,tests,190000);assert all(v['ok'] for v in checks),(arm,checks)
 rows=[local_eval.time_shape(problem,[side],case,2,1,timer) for case in benches];assert all(v['ok'] for v in rows),(arm,rows)
 result[arm]=dict(tests=checks,shapes=rows);print('PASS',arm,len(checks),len(rows),flush=True)
out=dict(forced=a.forced,manifest=json.loads((H/'manifest.json').read_text()),shared_prepare=True,elapsed_s=time.monotonic()-started,arms=result)
(H/('screen_forced.json' if a.forced else 'screen_normal.json')).write_text(json.dumps(out,indent=2)+'\n')
