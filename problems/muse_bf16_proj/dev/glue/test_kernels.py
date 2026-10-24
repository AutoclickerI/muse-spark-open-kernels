from pathlib import Path
import importlib.util
import argparse,json,time
import torch
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
path=Path(__file__).with_name('projection_glue.py')
spec=importlib.util.spec_from_file_location('glue',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
torch.manual_seed(32984)
weights={n:(torch.randn(a,b,device='cuda')/(b**.5)).to(dt) for n,a,b,dt in (
    ('qkv_gate',2304,8192,torch.bfloat16),('o',8192,1024,torch.bfloat16),
    ('pooler',4096,8192,torch.bfloat16),('unpooler',8192,4096,torch.bfloat16),
    ('router',256,8192,torch.float32))}
st=g.gb_state(weights);rows=[];start=time.monotonic();checks=0
for m in (1,8,16,32,128,256):
 for phase in ('b1','pool','b2'):
    x=torch.randn(m,8192 if phase=='pool' else 1024 if phase=='b1' else 4096,device='cuda',dtype=torch.bfloat16)
    stream=torch.randn(m,8192,device='cuda');data={'local_start':torch.tensor(19,device='cuda',dtype=torch.int32)}
    choices=list(g.gb_choices(m,phase)) if args.full else [(min(m,8),64,2 if phase=='b1' else 8,64 if phase=='pool' else 128,4)]
    refs,meta=g.gb_reference(phase,x,stream,st,data)
    for c in choices:
        plan=g.GbPlan(st,m,phase,c)
        actual=plan.run(x,stream,data);err=g.gb_check(phase,actual,refs,meta,stream,st,data)
        with g._PgGraph(lambda:plan.run(x,stream,data)) as graph:
            for repeat in range(3):
                x.normal_();stream.normal_();data['local_start'].fill_((19+repeat*31)%256)
                graph.graph.replay();refs,meta=g.gb_reference(phase,x,stream,st,data)
                err=max(err,g.gb_check(phase,graph.out,refs,meta,stream,st,data));checks+=1
                assert torch.equal(plan.counter,torch.zeros_like(plan.counter))
        row={'m':m,'phase':phase,'choice':c,'rel':err}
        rows.append(row);print(row,flush=True)
print('PASS',len(rows),'configs',checks,'fresh graphs','seconds',time.monotonic()-start,flush=True)
Path('runs/screen/verify_proj/glue_kernels.json').write_text(json.dumps({'rows':rows,'checks':checks,'elapsed':time.monotonic()-start},indent=2)+'\n')
