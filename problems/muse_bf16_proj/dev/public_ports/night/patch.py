"""Exact new per-site patch onto S21 + the selected native3347492 portfolio."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent
def apply(source,parent,groups,pool32_coded=False):
 sp=importlib.util.spec_from_file_location('ng_parent_portfolio_patch',H/'portfolio.py');p=importlib.util.module_from_spec(sp);sp.loader.exec_module(p);assert set(groups)<=set(('Q16','Q32','O16','P16','P32','U16','U32'));out=p.apply(source,parent);body=(H/'body.py.inc').read_text();out+='\n'+body+'\nNG_ENABLED='+repr(tuple(sorted(groups)))+'\nNG_POOL32_CODED='+repr(bool(pool32_coded))+'''
_ng_base_project=_best7_project
_ng_base_prepare=prepare
def _best7_project(x,role,state,plan,out_dtype=torch.bfloat16):
    key=(x.shape[0],role)
    if any(key in NG_GROUPS[g] for g in NG_ENABLED):return ng_project(x,role,state['night_gap'],pool32_coded=NG_POOL32_CODED)
    return _ng_base_project(x,role,state,plan,out_dtype)
def prepare(weights):
    global _best7_project
    selected=_best7_project;_best7_project=_ng_base_project
    try:state=_ng_base_prepare(weights)
    finally:_best7_project=selected
    bank=ng_prepare(globals(),state);start=time.monotonic()
    for key in sorted({k for g in NG_ENABLED for k in NG_GROUPS[g]}):
        t,role=key;w=bank['roles'][role];ng_project(torch.zeros((t,w.shape[1]),device=w.device,dtype=torch.bfloat16),role,bank,pool32_coded=NG_POOL32_CODED)
    torch.cuda.synchronize();bank['warm_s']=time.monotonic()-start;state['night_gap']=bank
    return state
'''
 compile(out,'<P10 night exact site patch>','exec');return out
if __name__=='__main__':
 source=Path(sys.argv[1]).read_text();parent=json.loads(Path(sys.argv[3]).read_text());groups=json.loads(Path(sys.argv[4]).read_text());out=apply(source,parent,groups);Path(sys.argv[2]).write_text(out);print(json.dumps(dict(base_sha256=hashlib.sha256(source.encode()).hexdigest(),sha256=hashlib.sha256(out.encode()).hexdigest(),groups=groups)))
