import time
import statistics
RI_DEADLINE=float('inf')


class RIBudget(RuntimeError):pass


def ri_need(seconds,where):
    if RI_DEADLINE-time.monotonic()<seconds:raise RIBudget(where)


def ri_plan(data,bank,stage):
    plan=wi_plan(data,bank,6,2112)
    if stage==0:return plan
    if stage not in (1,2):raise ValueError('route stage')
    if plan['mode']!=1 or FX_CONFIGS[plan['which']][3]!=1:raise ValueError('live route requires MODE1/SPLIT1')
    v=plan['v2'];m=plan['M'];cap=plan['cap'];start=plan['start'];clear=cap//32*193+m
    rp=[v[i].data_ptr() for i in (11,4,5,6,7,8,13,18,19)]
    def route():
        _=v
        bank['mod'].ri_route(rp,m,32,start,clear,stage,torch.cuda.current_stream().cuda_stream)
    plan['steps']=(route,)+plan['steps'][1:];return plan


def ri_call(data,bank,stage):
    for fn in ri_plan(data,bank,stage)['steps']:out=fn()
    return out


def ri_error(out,want):
    if not bool(torch.isfinite(out).all()):return float('inf'),float('inf')
    diff=out.float()-want.float();den=want.float().norm(dim=-1)
    return float(diff.norm()/want.float().norm().clamp_min(1e-12)),float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())


def ri_scratch(plan):
    unique={}
    for v in (plan['v13'],plan['v2']):
        for i in (4,5,6,7,8,9,10,13,15,16,17,18,19):
            t=v[i];unique[t.data_ptr()]=t
    return list(unique.values())


def ri_poison(plan,gen):
    for t in ri_scratch(plan):t.view(torch.uint8).random_(0,256,generator=gen)
