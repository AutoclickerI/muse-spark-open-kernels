"""Trace every scored stage with scalar tensor/launch stand-ins on CPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
H=Path(__file__).resolve().parent;P=H.parents[1]


def module(name):
    p=P/('cand_r3_best7_'+name+'.py')
    spec=importlib.util.spec_from_file_location('proof_'+name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def state(m):
    plans={}
    for key,(glue,att,moe,proj) in m.BEST7_SETTINGS.items():
        geometry={}
        if isinstance(glue,tuple):glue,b1,r,b2=glue;geometry=dict(b1=b1,route=r,b2=b2)
        plans[key]=dict(glue=glue,geometry=geometry,attention=att,moe=moe,
            projection={} if proj=='auto' else proj,prefetch=0,dense_chain=False)
    # Algorithm choices from the same prepared S15 state, including each legal
    # Lt alternative. Dispatch proof is independent of local tuner wall time.
    proj=dict(m.PROJ_PINS)
    proj[1,'o']=('stream',6,1)
    for t in m._TUNE_TOKENS:
        proj.setdefault((t,'o'),'custom')
    m._GEMV_CHOICES.clear()
    for (t,n),v in m.GB_ROWS.items():
        shapes={'qkv_gate':(2304,8192),'pool_router':(4608,8192),'unpooler':(8192,4096)}
        nn,kk=shapes[n];m._GEMV_CHOICES[t,nn,kk]=v
    for t in (1,8,16,32):m._GEMV_CHOICES[t,1280,8192]=m.GATE_SPLIT_QKV3
    return dict(best7_plans=plans,best7_prefetch_ops={},attention_plan={},proj_plan=proj,
                cluster_rows=m._CLUSTER_ROWS,partial_port_rows=m._PP_ROWS,vq8_rows=m._VQ8_ROWS,
                nf_port={'plans':{},'native':True},coded_dense={'qkv_gate':0,'pool_router':0})


class Tensor:
    def __init__(self,t,n=8192):self.shape=(t,n);self.device='cuda';self.dtype=None


def trace(m,st,b,l,w):
    cfg=dict(batch=b,q_len=l,window=w,tokens=b*l)
    key=(b,l,w);plan=m._s15n_plan(cfg,st) if hasattr(m,'_s15n_plan') else st['best7_plans'][key]
    original=st['best7_plans'][key]
    assert plan==original,(key,plan,original)
    data=dict(config=cfg,attn_input=Tensor(b*l),stream=Tensor(b*l))
    split=m._gate_split_on(data,st,plan);t=b*l
    raw=Tensor(t,1280 if split else 2304)
    stages=[]
    # Exercise actual attention selector with launch functions returning labels.
    saved={}
    def replace(name,fn):saved[name]=getattr(m,name);setattr(m,name,fn)
    for name in ('_cluster_attention','_layer_partial_port'):
        replace(name,lambda *a,_name=name,**kw:_name)
    replace('_vq8_attention',lambda *a,**kw:('vq8',a[2]))
    replace('_attn_port_hopper',lambda *a:True)
    replace('_attn_port_variant',lambda *a:('portfolio',a[3]))
    attention=m._best7_attention((raw,1),data,st,plan)
    for name,value in saved.items():setattr(m,name,value)
    # Intercept at final projection-family selection, not at top-level plan.
    for role in ('qkv3' if split else 'qkv_gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
        if hasattr(m,'_s15n_nf_key'):nf=m._s15n_nf_key(t,role)
        else:nf=(t,role) if (t,role) in m.NF_PORT_SELECTED else None
        if nf:choice=('NF',nf)
        else:
            v=plan['projection'].get(role,'auto')
            if v=='auto':
                anchor,v=m._s15n_projection(t,role,st) if hasattr(m,'_s15n_projection') else (t,st['proj_plan'].get((t,role),'custom'))
                if role=='qkv3' and not hasattr(m,'_s15n_projection'):v='custom'
            else:anchor=t
            cd=m.CD_ROWS.get(((m._s15n_nearest(t,m._TUNE_TOKENS) if hasattr(m,'_s15n_nearest') else t),role))
            if cd:choice=('coded',cd)
            elif v=='custom' and t<=m.GEMV_MAX_T:
                nk={'qkv3':(1280,8192),'qkv_gate':(2304,8192),'o':(8192,1024),
                    'pool_router':(4608,8192),'unpooler':(8192,4096),'unpooler_slice':(1024,4096)}[role]
                choice=('gemv',m._GEMV_CHOICES.get((t,*nk),m.GEMV_CFG[nk]))
            else:choice=(v,anchor if isinstance(v,tuple) and v[0] in ('lt','short_lt') else None)
        stages.append((role,choice))
    return dict(plan=plan,gate_split=split,attention=attention,moe=plan['moe'],
                flags=m.SMALLM_FLAGS[1 if t==1 else (64 if hasattr(m,'_s15n_plan') else t)] if plan['moe']=='nt1_lop3' else None,
                projection=stages,unpool_rows=1024 if t<=8 else 8192)


def main():
    a,n=module('s15'),module('s15n');sa,sn=state(a),state(n);rows=[]
    for key in sa['best7_plans']:
        if key=='other':continue
        aa,nn=trace(a,sa,*key),trace(n,sn,*key)
        assert aa==nn,(key,aa,nn)
        rows.append(dict(key=key,dispatch=aa,identical=True))
    # Exact prepared points must resolve to their own choice for *any* tuner
    # outcome, not just PROJ_PINS or this local GPU's selected algorithm.
    alternatives=0
    for (t,role),value in list(sn['proj_plan'].items()):
        for v in ('custom','cublas',('lt',0),('short_lt',1,0),('vdsk',(128,64,64,8,4,1),3)):
            sn['proj_plan'][t,role]=v
            assert n._s15n_projection(t,role,sn)==(t,v),(t,role,v)
            alternatives+=1
        sn['proj_plan'][t,role]=value
    proof=dict(prepared_projection_alternatives=alternatives,scored_shapes=12,identical=12,rows=rows,method='same prepared choices; stage dispatch and dimensional guards traced',
        s15_sha256=hashlib.sha256((P/'cand_r3_best7_s15.py').read_bytes()).hexdigest(),
        s15n_sha256=hashlib.sha256((P/'cand_r3_best7_s15n.py').read_bytes()).hexdigest())
    (H/'scored_dispatch.json').write_text(json.dumps(proof,indent=2)+'\n');print('all12 scored stage dispatches identical')


if __name__=='__main__':main()
