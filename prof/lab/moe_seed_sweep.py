"""Local GPU0 study with public generate_weights/weights_fields and real layer-front routing.
Every variant sees identical packed weights, activations and routes within a run seed.
No public benchmark currently has router_load targets: fields is {} unless upstream adds them.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'problems/muse_decode_layer'),str(ROOT/'tools')]
import local_eval as ev


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(ref,w,case,seed):
    data=ref.generate_input(**dict(case,seed=seed,local_start=0))
    f=ref.ref_front(data,w)
    # This is the actual layer's FP64 routing oracle, not the approximate tuner fixture.
    scores,biased=ref.route(f['h'],w);ids=biased.topk(8,-1).indices
    picked=scores.gather(1,ids);tw=picked/picked.sum(-1,keepdim=True)
    start=32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax())
    local=ids-start;local=torch.where((local>=0)&(local<32),local,-1).int()
    x=ref._rms(f['h']@w['pooler'].float().t())*(1+w['pre_expert_norm'].float())
    return dict(hidden_states=x.bfloat16(),topk_ids=local,topk_weights=tw.float(),config=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096))


def graph(fn,data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn(static)
    return static,g,out


def compare(paths,data,flush,reps):
    gs={name:graph(fn,data) for name,fn in paths.items()}
    for g in gs.values():
        for _ in range(10):g[1].replay()
    torch.cuda.synchronize()
    want=gs['NT1'][2].float();errs={}
    for name,g in gs.items():
        err=float((g[2].float()-want).norm()/want.norm().clamp_min(1e-12));assert err<.01,(name,err);errs[name]=err
    names=list(gs);vals={n:[] for n in names};a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for i in range(reps):
        order=names[i%len(names):]+names[:i%len(names)]
        if i%2:order=order[::-1]
        for name in order:
            flush.zero_();a.record();gs[name][1].replay();b.record();b.synchronize();vals[name].append(a.elapsed_time(b)*1000)
    return {n:dict(us=statistics.median(v),relative_l2=errs[n]) for n,v in vals.items()}


def summarize(rows):
    out=[]
    for m in (128,256):
        for name in rows[0]['timings']:
            rr=[r for r in rows if r['M']==m];v=[r['timings'][name]['us'] for r in rr]
            out.append(dict(M=m,kernel=name,mean_us=statistics.mean(v),min_us=min(v),max_us=max(v),max_min=max(v)/min(v),max_mean=max(v)/statistics.mean(v),seeds=len(v)))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=12);ap.add_argument('--reps',type=int,default=8);ap.add_argument('--balanced',action='store_true');ap.add_argument('--hot',action='store_true');ap.add_argument('--split',action='store_true');ap.add_argument('--tune',action='store_true');ap.add_argument('--cache',type=Path,default=ROOT/'runs/screen/v9_seed/cache.pt');ap.add_argument('--out',type=Path,default=ROOT/'runs/screen/v9_seed/baseline.json');args=ap.parse_args()
    ref=module('seed_reference',ROOT/'problems/muse_decode_layer/reference.py')
    if hasattr(ref,'_fa3_scheduler_metadata'):ref._fa3_scheduler_metadata=lambda *a,**kw:None
    d=module('seed_dropin',ROOT/('runs/screen/v9_seed/tune_dropin.py' if args.tune else 'problems/muse_moe_w4a16/dev/v9_reuse/dropin.py'))
    bal=module('seed_balanced',ROOT/('problems/muse_moe_w4a16/dev/v9_reuse/split_w2.py' if args.split else 'problems/muse_moe_w4a16/dev/v9_reuse/balanced.py')) if args.balanced or args.split else None
    hot=module('seed_hot',ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/hot_v9.py') if args.hot else None
    cases=json.loads((ROOT/'upstream/api/problems/muse_decode_layer.json').read_text())['benchmarks']
    targets=[c for c in cases if c['load'] in ('verify 16','verify 32') and c['layer']=='sliding']
    cache=torch.load(args.cache,weights_only=False) if args.cache.exists() else {}
    flush=torch.empty(128<<20,device='cuda',dtype=torch.uint8);rows=[]
    for i in range(args.seeds):
        seed=5353+7919*i;fields_seed=104729+100003*i;t0=time.monotonic()
        w=ref.generate_weights(seed)
        fields=ref.weights_fields(w,cases,fields_seed)
        if fields:w=ref.generate_weights(seed,**fields)
        w['post_norm_w']=w['post_expert_norm'];bank=d.rv_prepare(w);rb=dict(bank,reuse_mod=d.rr_reuse_module())
        if bal:bb=bal.balanced_prepare(bank)
        if hot:hb=dict(bank,mod=hot._module())
        for case in targets:
            m=case['batch']*case['q_len'];key=f'{seed}:{m}'
            if key in cache:
                data={k:v.cuda() if torch.is_tensor(v) else v for k,v in cache[key]['data'].items()}
            else:
                data=fixture(ref,w,case,case['seed']+i*0x2545F4914F6CDD1D%(2**63-1))
                cache[key]=dict(seed=seed,fields_seed=fields_seed,fields=fields,data={k:v.cpu() if torch.is_tensor(v) else v for k,v in data.items()})
            counts=torch.bincount(data['topk_ids'][data['topk_ids']>=0],minlength=32)
            paths={f'NT{nt}':lambda x,nt=nt:d.rv_kernel_fn(x,dict(bank,nt=nt)) for nt in (1,2,4)}
            paths.update(hot_r32=lambda x:d.rr_reuse_call(x,rb,(4,0,2)),hot_s128=lambda x:d.rr_reuse_call(x,rb,(16,1,2)))
            if bal:
                for choice in bal.BAL_CHOICES:paths[bal.balanced_label(choice)]=lambda x,c=choice:bal.balanced_call(x,bb,c)
            if args.tune:
                for nt in (1,2,4):
                    for stages in (1,2):
                        for factor in (1,2):
                            paths[f'N{nt}s{stages}g{factor}']=lambda x,n=nt,s=stages,f=factor:d.rv_kernel_fn(x,dict(bank,nt=n,stage_override=s,grid_override=f))
            if hot:
                for nt,ch,st in ((1,0,2),(2,0,2),(2,0,1),(2,16,1),(2,32,1),(2,64,1),(2,32,2),(2,64,2)):
                    paths[f'hotNT{nt}c{ch}s{st}']=lambda x,n=nt,c=ch,st=st:hot.kernel_fn(x,dict(hb,nt=n,sched=1,chunk_cap=c,stages=st))
            timings=compare(paths,data,flush,args.reps)
            row=dict(seed=seed,fields_seed=fields_seed,fields=fields,M=m,case=case,counts=counts.cpu().tolist(),pairs=int(counts.sum()),experts=int((counts>0).sum()),hot=int(counts.max()),timings=timings)
            rows.append(row);print(json.dumps(row),flush=True)
        torch.save(cache,args.cache)
        print('SEED_DONE',i,seed,round(time.monotonic()-t0,2),flush=True)
    summary=summarize(rows);out=dict(gpu=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fields_note='Public benchmarks omit router_load: weights_fields returns {}; no hidden targets invented.',rows=rows,summary=summary)
    args.out.write_text(json.dumps(out,indent=2)+'\n')
    print('SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
