"""Apply explicit P10 port points to the current S21/S22 projection ABI.

python3 patch.py BASE.py OUTPUT.py POINTS.json
POINTS: [[T, role, provider], ...]. Whole-layer G0 remains P4's gate.
"""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent

def apply(source,points):
    selected={(t,r):p for t,r,p in points}
    assert len(selected)==len(points) and all(p in ('O','P','C','N') for p in selected.values())
    helper=(H/'port_body.py.inc').read_text()
    tail='''
PP_SELECTED=SELECTION
_pp_original_project=_best7_project
_pp_original_gate=_gate_split_apply
_pp_original_prepare=prepare
def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    provider=PP_SELECTED.get((x.shape[0],name))
    if provider is not None:return pp_project(x,name,state['p10_public'],provider)
    return _pp_original_project(x,name,state,plan,out_dtype)
def _gate_split_apply(x,y,state):
    provider=PP_SELECTED.get((x.shape[0],'gate'))
    if provider is not None:return pp_gate(x,y,state['p10_public'],provider)
    return _pp_original_gate(x,y,state)
def prepare(weights):
    global _best7_project,_gate_split_apply
    selected_project,selected_gate=_best7_project,_gate_split_apply
    _best7_project,_gate_split_apply=_pp_original_project,_pp_original_gate
    try:state=_pp_original_prepare(weights)
    finally:_best7_project,_gate_split_apply=selected_project,selected_gate
    bank=pp_prepare(globals(),weights,state,{(t,r,p) for (t,r),p in PP_SELECTED.items()})
    start=time.monotonic()
    for (t,r),p in PP_SELECTED.items():
        w=bank['roles'][r];x=torch.zeros((t,w.shape[1]),device=w.device,dtype=torch.bfloat16)
        if r=='gate':pp_gate(x,torch.zeros((t,1024),device=w.device,dtype=torch.bfloat16),bank,p)
        else:pp_project(x,r,bank,p)
    torch.cuda.synchronize();bank['warm_s']=time.monotonic()-start
    state['p10_public']=bank
    return state
'''.replace('SELECTION',repr(selected))
    out=source+'\n'+helper+tail;compile(out,'<P10 explicit projection port>','exec');return out

if __name__=='__main__':
    base=Path(sys.argv[1]).read_text();points=json.loads(Path(sys.argv[3]).read_text());out=apply(base,points);Path(sys.argv[2]).write_text(out)
    print(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),source_sha256=hashlib.sha256(out.encode()).hexdigest(),points=points)))
