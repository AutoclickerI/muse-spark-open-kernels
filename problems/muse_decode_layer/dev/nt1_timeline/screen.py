"""Local screen (sm_120, loose): router+MoE graphs at M (M=..., default 1), paired AB/BA vs INC, read flush; plus the traced
critical path of each arm on one seed. Usage: screen.py [M=32] [SEEDS=16] flags ..."""
import os,sys,time,statistics,linecache,torch
kv=dict(a.split('=') for a in sys.argv[1:] if '=' in a);M=int(kv.get('M',1));SEEDS=int(kv.get('SEEDS',16))
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
_buf=[]
def lab_cold_l2():
    if not _buf:_buf.append(torch.empty(1<<30,dtype=torch.uint8,device='cuda'))
    _buf[0].view(torch.int64).sum()
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();n={'__name__':'h200lab','lab_cold_l2':lab_cold_l2};f='<lab>'
linecache.cache[f]=(len(src),None,src.splitlines(True),f);exec(compile(src,f,'exec'),n)
flags=[int(x) for x in sys.argv[1:] if '=' not in x];deltas={x:[] for x in flags};base=[]
for seed in range(SEEDS):
    w=n['rg_generate_weights'](5353+7919*seed);w['post_norm_w']=w['post_expert_norm'];state=n['nt1_banks'](w)
    pr,rs=n['routed_fixture'](w,M,9001+97*seed)
    g0=n['TimelineGraph'](lambda:n['routed_experts'](pr,rs,state,'INC'))
    for x in flags:
        g=n['TimelineGraph'](lambda x=x:n['routed_var'](pr,rs,state,x))
        a,b,r=n['timeline_timing'](g0,g);deltas[x].append(a*(r-1))
    base.append(a)
    if seed==0:
        d=n['fx_make_fixture'](w,M,9001)
        for x in flags:
            n['DIAG_FLAGS']['RTRACE%d'%x]=x|256
        res=n['timeline_diagnose'](d,state,n['routed_fixture'](w,M,9001),tuple('RTRACE%d'%x for x in flags))
        for l in n['timeline_lines'](res,False):
            if l.startswith(('CRITICAL','SKEW','ACCOUNT','TILE')):print(l[:400])
print('RINC',round(statistics.mean(base),2))
for x,v in deltas.items():print(x,f'{statistics.mean(v):+.2f} +- {statistics.stdev(v)/len(v)**.5:.2f} n{len(v)}')
