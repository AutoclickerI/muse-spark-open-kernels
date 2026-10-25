"""local_eval normal/forced screens for exact-CTA b1-only and b2-only ablations, one prepare."""
import argparse,hashlib,importlib.util,json,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];PDIR=ROOT/'problems/muse_decode_layer'
sys.path[:0]=[str(ROOT/'tools'),str(PDIR),str(HERE)]
import local_eval
from screen import replay_us

def main():
 p=argparse.ArgumentParser();p.add_argument('--forced',action='store_true');a=p.parse_args();local_eval.Timer.replay_us=replay_us
 ref=local_eval.load_module('reference',PDIR/'reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
 m=local_eval.load_module('br_role',HERE/'exact_trial.py')
 if a.forced:
  sp=importlib.util.spec_from_file_location('br_role_force',PDIR/'dev/best7/force.py');force=importlib.util.module_from_spec(sp);sp.loader.exec_module(force);force.install(m,False,False)
 weights=ref.generate_weights(ref.WEIGHTS_SEED);state=m.prepare(weights);problem=local_eval.Problem(ref,weights);tests=local_eval.tests_of((PDIR/'TASK.md').read_text());benches=json.loads((ROOT/'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks'];timer=local_eval.Timer();result={}
 for label,role in [('E1','b1'),('E2','b2')]:
  st=dict(state,best7_plans={k:dict(p) for k,p in state['best7_plans'].items()})
  for key,plan in st['best7_plans'].items():
   if key!='other' and key[0] in (16,32) and key[1] in (1,8):plan['br_roles']=(role,)
  side=types.SimpleNamespace(call=lambda data,st=st:m.kernel_fn(data,st));checks=local_eval.run_tests(problem,side,tests,90000);rows=[local_eval.time_shape(problem,[side],case,2,1,timer) for case in benches]
  assert all(r['ok'] for r in checks+rows),(label,checks,rows);result[label]=dict(tests=checks,shapes=rows);print('PASS',label,len(checks),'tests',len(rows),'captured shapes',flush=True)
 (HERE/('roles_forced.json' if a.forced else 'roles_normal.json')).write_text(json.dumps(dict(source_sha256=hashlib.sha256((HERE/'exact_trial.py').read_bytes()).hexdigest(),forced=a.forced,arms=result),indent=2)+'\n')
if __name__=='__main__':main()
