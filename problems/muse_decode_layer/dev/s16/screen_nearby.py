"""Independent TASK checks for composed S16 on all 65 review-shaped cases."""
import argparse,gc,hashlib,json,sys,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
import local_eval as ev,screen


def main():
    ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);ap.add_argument('--forced',action='store_true');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cases=screen.load(H.parent/'nearby/cases.py','s16_nearby_case_set');original=ev.load_module;rows=[]
    def loader(name,path):
        m=original(name,path)
        if name=='reference':m.CACHE_CHUNK=8192
        if name=='submission':screen.install(m,a.forced)
        return m
    ev.load_module=loader
    def run_tests(problem,side,official,seed0):
        todo=[(dict(c,label='TASK-'+str(i)),seed0+i) for i,c in enumerate(official)]
        todo += [(dict(c),seed) for c in cases.cases() for seed in cases.SEEDS]
        for c,seed in todo:
            label=c.pop('label');row=dict(label=label,seed=seed)
            data=out=graph=None
            try:
                data=problem.ref.generate_input(**c,seed=seed)
                front=problem.ref.ref_front(data,problem.weights);ids=front['biased'].topk(8,dim=-1).indices
                start=32*int(ev.torch.bincount((ids//32).flatten(),minlength=8).argmax());data['local_start'].fill_(start);del front,ids
                out=side.call(data);ev.torch.cuda.synchronize();row['eager_ok'],row['eager_msg']=problem.check(data,out)
                graph,out=ev.capture(side.call,data);graph.replay();graph.replay();ev.torch.cuda.synchronize()
                row['graph_ok'],row['graph_msg']=problem.check(data,out);row['ok']=row['eager_ok'] and row['graph_ok']
            except Exception as exc:row.update(ok=False,error=type(exc).__name__+':'+str(exc));traceback.print_exc()
            rows.append(row);print('S16_NEARBY',label,seed,row['ok'],flush=True)
            data=out=graph=None;gc.collect();ev.torch.cuda.empty_cache()
        return [dict(case=r['label']+':'+str(r['seed']),ok=r['ok'],msg=r.get('error',r.get('graph_msg',''))) for r in rows]
    ev.run_tests=run_tests;a.out.mkdir(parents=True,exist_ok=True)
    sys.argv=['local_eval.py',str(a.candidate),'--tests-only','--json',str(a.out/'local.json')]
    try:ev.main()
    finally:
        result=dict(sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),cases=len(rows),passed=len(rows)==143 and all(r['ok'] for r in rows),tests=rows)
        (a.out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
