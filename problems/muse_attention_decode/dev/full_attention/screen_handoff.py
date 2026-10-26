"""Whole-layer local_eval of an exact S15 attention handoff, with graph replay."""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--forced',action='store_true')
ap.add_argument('--capacity-boundary',action='store_true');a=ap.parse_args()
candidate=a.candidate.resolve();mode='forced' if a.forced else 'normal';holder={};original=ev.load_module
def loader(name,path):
    m=original(name,path)
    if name=='reference':m.CACHE_CHUNK=8192
    if name=='submission':
        if a.forced:
            spec=importlib.util.spec_from_file_location('handoff_force',R/'problems/muse_decode_layer/dev/verify_best.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False)
        import local_standins
        local_standins.install(m);holder['module']=m
    return m
ev.load_module=loader
old_tests=ev.tests_of
def tests(md):
    extra=[dict(batch=b,q_len=8,ctx=129,layer='full',page=1,dist='soft',spread='fixed') for b in (64,65)] if a.capacity_boundary else []
    return (old_tests(md)+[dict(batch=b,q_len=q,ctx=80000,layer=layer,page=1,dist='soft',
        spread='agentic' if b>1 else 'fixed') for b in (1,16,32) for q in (1,8) for layer in ('sliding','full')]+extra)
ev.tests_of=tests
def run(problem,side,cases,seed0):
    rows=[]
    for i,c in enumerate(cases):
        row=dict(case=ev.describe(c));d=problem.make_input(c,seed0+i)
        try:
            y=side.call(d);ev.torch.cuda.synchronize();ok,msg=problem.check(d,y)
            row.update(eager_ok=ok,eager_msg=msg)
            g,y=ev.capture(side.call,d);g.replay();g.replay();ev.torch.cuda.synchronize()
            ok,msg=problem.check(d,y);row.update(replay_ok=ok,msg=msg,ok=row['eager_ok'] and ok)
            del g,y
        except Exception as e:row.update(ok=False,msg=type(e).__name__+':'+str(e))
        rows.append(row);print('HANDOFF',mode,row,flush=True);del d;gc.collect();ev.torch.cuda.empty_cache()
    return rows
ev.run_tests=run
report=candidate.with_name(candidate.stem+'_'+mode+'_screen.json')
sys.argv=['local_eval.py',str(candidate),'--tests-only','--json',str(report)]
ev.main()
