"""Official local_eval checks on nearby shapes; no candidate edits."""
import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'nearby'),str(H.parent/'s15')]
import local_eval as ev
from cases import S15_SHA, SEEDS, cases


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true')
    ap.add_argument('--wrapper',type=Path);ap.add_argument('--only',nargs='*')
    ap.add_argument('--candidate',type=Path)
    a=ap.parse_args();mode='forced' if a.forced else 'normal'
    candidate=a.candidate.resolve() if a.candidate else R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
    if a.candidate:mode='s15e_'+mode
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    result=dict(mode=mode,candidate_sha256=digest,seeds=SEEDS,tests=[],dispatch={},complete=False)
    original_loader=ev.load_module;holder={}
    def loader(name,path):
        m=original_loader(name,path)
        if name=='reference':
            # Setup scratch only. All context keys and the TASK check are kept.
            m.CACHE_CHUNK=8192
        if name=='submission':
            holder['module']=m
            if a.forced:
                spec=importlib.util.spec_from_file_location('nearby_force',H.parent/'verify_best.py')
                helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
                helper.install(m,False)
            import local_standins
            result['standins']=local_standins.install(m)
            for label in ('_best7_attention','_best7_expert','_best7_project'):
                fn=getattr(m,label)
                def trace(*args,_fn=fn,_label=label,**kw):
                    if _label=='_best7_project':key=f'project:T{args[0].shape[0]}:{args[1]}'
                    elif _label=='_best7_attention':
                        c=args[1]['config'];key=f'attention:B{c["batch"]}:Q{c["q_len"]}:W{c["window"]}'
                    else:key=f'moe:T{args[1].shape[0]}:{args[-1]["moe"]}'
                    counts=result['dispatch'].setdefault(key,dict(eager=0,captured=0))
                    counts['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
                    return _fn(*args,**kw)
                setattr(m,label,trace)
        return m
    ev.load_module=loader
    def run_tests(problem,side,official,seed0):
        result['gpu']=ev.torch.cuda.get_device_name();m=holder['module']
        chosen=cases()
        if a.only:chosen=[c for c in chosen if c['label'] in a.only]
        todo=[(dict(c),s) for c in chosen for s in SEEDS]
        if not a.only:todo=[(dict(c,label=f'TASK-{i}'),seed0+i) for i,c in enumerate(official)]+todo
        result['plans']={str(k):v for k,v in side.state['best7_plans'].items()} if hasattr(side,'state') else {}
        for c,seed in todo:
            label=c.pop('label');row=dict(label=label,case=c,seed=seed)
            data=out=graph=None
            try:
                # Same busiest-range definition as case_fields, without making
                # a second multi-GB cache just to generate this input twice.
                data=problem.ref.generate_input(**c,seed=seed)
                front=problem.ref.ref_front(data,problem.weights)
                ids=front['biased'].topk(8,dim=-1).indices
                start=32*int(ev.torch.bincount((ids//32).flatten(),minlength=8).argmax())
                data['local_start'].fill_(start);del front,ids
                row['seq_range']=[int(data['seq_lens'].min()),int(data['seq_lens'].max())]
                out=side.call(data);ev.torch.cuda.synchronize()
                row['eager_ok'],row['eager_msg']=problem.check(data,out)
                # Capture and replay twice. Each seed independently exercises
                # its live routing, outputs and newly written cache slots.
                graph,out=ev.capture(side.call,data)
                graph.replay();graph.replay();ev.torch.cuda.synchronize()
                row['graph_ok'],row['graph_msg']=problem.check(data,out)
                row['ok']=row['eager_ok'] and row['graph_ok']
            except Exception as exc:
                row.update(ok=False,error=type(exc).__name__+': '+str(exc))
                traceback.print_exc()
            result['tests'].append(row)
            print('NEARBY',mode,label,seed,'PASS' if row['ok'] else 'FAIL',row.get('error',row.get('graph_msg','')),flush=True)
            (H/f'{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
            del graph,out,data;gc.collect();ev.torch.cuda.empty_cache()
        result['complete']=len(result['tests'])==len(todo)
        return [dict(case=r['label']+':'+str(r['seed']),ok=r['ok'],msg=r.get('error',r.get('graph_msg',''))) for r in result['tests']]
    ev.run_tests=run_tests
    target=a.wrapper.resolve() if a.wrapper else candidate
    if not a.wrapper and not a.candidate:assert hashlib.sha256(target.read_bytes()).hexdigest()==S15_SHA
    sys.argv=['local_eval.py',str(target),'--tests-only','--json',str(H/f'{mode}_local_eval.json')]
    try:ev.main()
    finally:(H/f'{mode}.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
