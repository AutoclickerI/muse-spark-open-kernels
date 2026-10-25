"""Fuse small-M boundary/route steps into their GEMV producers on S11 final.

Experimental roles are explicit in PG_ROLES. Empty by default until paired
whole-layer evidence. --trial admits all three producer epilogues locally.
"""
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROLES=()
BLOCK='''

def _pg_prepare(state,device):
    state['pg_ops']={}
    for t in range(1,33):
        for role,name,s in (('b1','o',2),('route','pool_router',6),
                            ('b2','unpooler_slice' if t<=8 else 'unpooler',8 if t<=8 else 2)):
            state['pg_ops'][(t,role)]=PG_PRODUCER(state[name],t,s,role)


def _pg_project_boundary(x,stream,state,plan,phase,sync=None):
    if phase in state.get('pg_roles',PG_ROLES) and (x.shape[0],phase) in state['pg_ops']:
        op=state['pg_ops'][(x.shape[0],phase)]
        result=op(x,state,stream)
        if not op.fuse:return _best7_boundary(result,stream,state,plan,phase,sync)
        return result
    name='o' if phase=='b1' else ('unpooler_slice' if x.shape[0]<=8 else 'unpooler')
    branch=_best7_project(x,name,state,plan)
    return _best7_boundary(branch,stream,state,plan,phase,sync)


def _pg_pool_project(h,state,plan):
    if 'route' in state.get('pg_roles',PG_ROLES) and (h.shape[0],'route') in state['pg_ops']:
        return h
    return _best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)


def _pg_route_stage(pr,state,data,plan,sync=None):
    t=data['config']['tokens']
    if 'route' in state.get('pg_roles',PG_ROLES) and (t,'route') in state['pg_ops']:
        op=state['pg_ops'][(t,'route')]
        out=op(pr,state,start=data['local_start'])
        if not op.fuse:return out,_best7_route(out,state,data,plan,sync)
        return (op.p,op.s),out
    return pr,_best7_route(pr,state,data,plan,sync)
'''


def patch(src,roles=None,stats=False,hybrid=False):
    stats=stats or hybrid
    roles=ROLES if roles is None else roles
    changes={
        "        a=_best7_project(attn,'o',state,plan)\n        h,stream1=_best7_boundary(a,data['stream'],state,plan,'b1',sync[0])":
        "        h,stream1=_pg_project_boundary(attn,data['stream'],state,plan,'b1',sync[0])",
        "        pr=_best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)":
        "        pr=_pg_pool_project(h,state,plan)",
        "    x,ids,wts,lids=_best7_route(pr,state,data,plan,sync[1])":
        "    pr,(x,ids,wts,lids)=_pg_route_stage(pr,state,data,plan,sync[1])",
        "    f=_best7_project(moe,name,state,plan)\n    hidden,stream2=_best7_boundary(f,stream1,state,plan,'b2',sync[2])":
        "    hidden,stream2=_pg_project_boundary(moe,stream1,state,plan,'b2',sync[2])",
        "    _gate_split_prepare(state,device)\n":
        "    _gate_split_prepare(state,device)\n    _pg_prepare(state,device)\n",
    }
    for old,new in changes.items():
        assert src.count(old)==1,(old[:80],src.count(old))
        src=src.replace(old,new)
    code=(HERE/'kernels.py').read_text()
    if stats:
        code+='\n'+(HERE/'stats_kernel.py').read_text().replace('from kernels import ProducerGlue, _pg_wait, _pg_clock, _pg_meta\n','')
    if hybrid:
        code+='\n'+(HERE/'fixed_o.py').read_text().replace('__CUDA__',repr((HERE/'fixed_o.cu').read_text()),1)
    return (src+'\n\n'+code+'\n\n'+BLOCK+'\nPG_ROLES = '+repr(tuple(roles))+
            '\nPG_PRODUCER = '+('ProducerHybrid' if hybrid else 'ProducerStats' if stats else 'ProducerGlue')+'\n')


if __name__=='__main__':
    out=patch(Path(sys.argv[1]).read_text(),('b1','route','b2') if '--trial' in sys.argv[3:] else ROLES,
              stats='--stats' in sys.argv[3:],hybrid='--hybrid' in sys.argv[3:])
    Path(sys.argv[2]).write_text(out)
    print(sys.argv[2],hashlib.sha256(out.encode()).hexdigest())
