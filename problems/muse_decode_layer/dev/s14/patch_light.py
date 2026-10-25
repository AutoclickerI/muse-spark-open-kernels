"""Exact eea9d87b N16G4: NT1 <=8, N16 light 9..16, PERMW heavy, third stream."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent

def once(s,a,b):
    assert s.count(a)==1,a[:140]
    return s.replace(a,b)

def patch(src):
    lines=src.splitlines(keepends=True);edits=[]
    for n in ast.parse(src).body:
        name=getattr(n,'name','')
        if isinstance(n,ast.Assign):name=getattr(n.targets[0],'id','')
        if name=='NF_CUDA':edits.append((n.lineno,n.end_lineno,'NF_CUDA='+repr((H/'light_cuda.cu').read_text())+'\n'))
        elif name=='nf_plan':edits.append((n.lineno,n.end_lineno,(H/'light_plan.py.inc').read_text()))
        elif name in ('NF128_CUDA','nf128_module'):
            first=min([d.lineno for d in getattr(n,'decorator_list',[])]+[n.lineno]);edits.append((first,n.end_lineno,''))
    assert len(edits)==4,edits
    for a,b,text in sorted(edits,reverse=True):lines[a-1:b]=[text]
    src=''.join(lines)
    src=once(src,'        nf128_future=native_pool.submit(nf128_module)\n','')
    src=once(src,'nf128_future.result();nf_future.result()','nf_future.result()')
    src=once(src,"state['s5_hh_base']=issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))\n    state['s13_permw_base']=nf_perm(nf_wide(dict(state['s5_hh_base'])))",
             "state['s5_hh_base']=nf_perm(nf_wide(issue_prepare(nf_ilv(dict(weights,post_norm_w=weights['post_expert_norm'])))))")
    src=once(src,"state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event())",
             "state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(device=device),torch.cuda.Event())")
    src=once(src,"state['s13_permw_base'] if data['hidden_states'].shape[0]>128 else state['s5_hh_base']","state['s5_hh_base']")
    compile(src,'<S14 N16G4>','exec');return src
