"""Align synthetic glue admission with published TASK routing semantics only."""
import ast,hashlib

HELPER='''
def _s16_prepare_check_routing(ids,biased,local_start):
    # TASK.md / reference.check_routing: bounds, unique picks, score/keep
    # margins, then local route and distinct-expert work. Preparation only.
    if ids.min()<0 or ids.max()>=256:raise ValueError('prepare routing: expert out of range')
    ranked=ids.long().sort(-1).values
    if (ranked[:,1:]==ranked[:,:-1]).any():raise ValueError('prepare routing: duplicate expert')
    best=biased.topk(8,dim=-1);gap=best.values[:,-1:]-biased.gather(1,ids.long())
    if gap.max()>.05:raise ValueError('prepare routing: selected score outside TASK margin')
    ninth=biased.topk(9,dim=-1).values[:,-1:]
    kept=(best.indices[:,:,None]==ids.long()[:,None,:]).any(-1)
    local=(best.indices-local_start)%256<32
    if (local & ~kept & (best.values-ninth>(.03 if len(ids)<=256 else .04))).any():
        raise ValueError('prepare routing: missed required local expert')
    actual=ids[(ids-local_start)%256<32];ref=best.indices[local]
    if actual.numel()<ref.numel()*.92-4:raise ValueError('prepare routing: insufficient local routes')
    if torch.unique(actual).numel()<torch.unique(ref).numel()*.85-2:raise ValueError('prepare routing: insufficient distinct local experts')
    if not torch.equal(ranked,best.indices.sort(-1).values):
        print('PREP_ROUTE_TASK_NEAR_TIE',len(ids),'max_score_gap',float(gap.max()),flush=True)
'''


def apply(source):
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_tune_glue'];assert len(nodes)==1
    n=nodes[0];lines=source.splitlines(True);old=''.join(lines[n.lineno-1:n.end_lineno]);new=old
    before='                want=_glue_oracle(project(x,state[name],name,state,out_dtype=dtype),stream,alpha,beta,prew,postw,bias)'
    assert new.count(before)==1
    new=new.replace(before,'''                value=project(x,state[name],name,state,out_dtype=dtype)
                want=_glue_oracle(value,stream,alpha,beta,prew,postw,bias)
                if route:
                    p,_=value
                    logits=(p[:,:,4096:4352].float()+p[:,:,4352:].float()).sum(0)
                    scores=(logits*ROUTER_MULT).sigmoid()
                    want=(want[0],scores+bias,scores)''')
    before='                    got=invoke(x,stream)\n';assert new.count(before)==1
    new=new.replace(before,before+'''                    if route:
                        norm,ids,tw,local=got
                        _s16_prepare_check_routing(ids,expected[1],243)
                        picked=expected[2].gather(1,ids.long())
                        loc=(ids-243+256)%256
                        expected=(expected[0],ids,picked/(picked.sum(-1,keepdim=True)+1.e-15),torch.where(loc<32,loc,-1).int())
''')
    lines[n.lineno-1:n.end_lineno]=[new];fixed=''.join(lines)+'\n'+HELPER
    a,b=ast.parse(source).body,ast.parse(fixed).body
    unchanged=lambda rows:[ast.dump(n) for n in rows if not isinstance(n,ast.FunctionDef) or n.name not in ('_tune_glue','_s16_prepare_check_routing')]
    assert unchanged(a)==unchanged(b),'timed source changed'
    return fixed


REVISION=hashlib.sha256(HELPER.encode()).hexdigest()
