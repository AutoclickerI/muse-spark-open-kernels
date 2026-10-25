"""Drop-in cell check: smallm_experts(flags) vs the admitted INC on the S6 bank layout, eager and graph, M=1/8/16/32."""
import runpy,linecache,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];DEV=Path(__file__).resolve().parent
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();n={};fname='<lab>';linecache.cache[fname]=(len(src),None,src.splitlines(True),fname)
exec(compile(src,fname,'exec'),n)
cell=runpy.run_path(str(DEV/'smallm_var_cell.py'))
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];lab_state=n['nt1_banks'](w)
s4=dict(gs13=lab_state['gs13'],gs2=lab_state['gs2'],pnw=lab_state['pnw'],sms=lab_state['sms'],
        counters={'lop3':torch.zeros(64,device='cuda',dtype=torch.int32)},banks={'lop3':dict(lab_state['bank'])})
choice={1:[95],8:[69],16:[69],32:[69]}
for m,fs in choice.items():
    for f in fs:cell['smallm_bind'](s4,{m:f})
worst=0.
for m,fs in choice.items():
    for seed in range(3):
        d=n['fx_make_fixture'](w,m,777+seed+m);want=n['nt1_fp32'](d,w);inc=n['nt1_call'](d,lab_state,'INC')
        for f in fs:
            fn=lambda:cell['smallm_experts'](d,s4,f);eager=fn();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            g.replay();torch.cuda.synchronize()
            for got in (eager,out):
                r=float((got.float()-inc.float()).norm()/inc.float().norm());ok,err,row=n['task_check'](got,want)
                assert ok and r<2e-3,(m,f,err,row,r);worst=max(worst,r)
print('cell OK worst rel vs INC',worst)
