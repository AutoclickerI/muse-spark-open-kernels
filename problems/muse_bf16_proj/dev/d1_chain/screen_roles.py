"""local_eval checks for W1/WR/W2 with one shared prepare; optional standard Hopper stand-ins."""
import argparse,hashlib,importlib.util,json,sys,time,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PDIR=ROOT/'problems/muse_decode_layer'
sys.path[:0]=[str(ROOT/'tools'),str(PDIR),str(HERE)]
import local_eval
from screen import replay_us

def main():
 p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args()
 local_eval.Timer.replay_us=replay_us
 ref=local_eval.load_module('reference',PDIR/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
 mod=local_eval.load_module('d1_role_submission',HERE/'weights_trial.py')
 if a.forced:
  sp=importlib.util.spec_from_file_location('d1_role_force',PDIR/'dev/best7/force.py');force=importlib.util.module_from_spec(sp);sp.loader.exec_module(force);force.install(mod,False,False)
 weights=ref.generate_weights(ref.WEIGHTS_SEED);started=time.monotonic();state=mod.prepare(weights)
 problem=local_eval.Problem(ref,weights);tests=local_eval.tests_of((PDIR/'TASK.md').read_text());benches=json.loads((ROOT/'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks'];timer=local_eval.Timer();results={}
 for label,role in [('W1','b1'),('WR','route'),('W2','b2')]:
  mod.GP_ROWS=tuple((1,1,w,role) for w in (2048,0));st=dict(state);side=types.SimpleNamespace(call=lambda data,st=st:mod.kernel_fn(data,st),prepare_s=0.)
  checks=local_eval.run_tests(problem,side,tests,90000);assert all(r['ok'] for r in checks),(label,checks)
  rows=[local_eval.time_shape(problem,[side],case,2,1,timer) for case in benches];assert all(r['ok'] for r in rows),(label,rows)
  results[label]=dict(role=role,tests=checks,shapes=rows);print('PASS',label,len(checks),'tests',len(rows),'captured shapes',flush=True)
 out=dict(weights_candidate_sha256=hashlib.sha256((HERE/'weights_trial.py').read_bytes()).hexdigest(),forced=a.forced,shared_prepare=True,elapsed_s=time.monotonic()-started,arms=results)
 (HERE/('roles_forced.json' if a.forced else 'roles_normal.json')).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
