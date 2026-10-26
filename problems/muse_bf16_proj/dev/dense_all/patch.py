"""Prune the measured coding recipe to selected (T,role) dispatches on a layer.

Usage: patch.py BASE.py OUTPUT.py selection.json
The JSON is a list of [T, role, arm]; gate_apply is a fused consumer.
"""
import ast,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent

def helper(selection):
    sys.path.insert(0,str(H))
    from recipe import helper_body
    body=helper_body();selection={tuple(k):int(v) for k,v in selection.items()}
    normalized={(t,'unpooler_slice' if r=='unpooler' and t<=8 else 'gate' if r=='gate_apply' else r):a for (t,r),a in selection.items()}
    roles={r for t,r in normalized};requests={(t,8192 if r in ('qkv_gate','qkv3','gate','pool_router') else 1024 if r=='o' else 4096,a) for (t,r),a in normalized.items() if t<=32}
    def swap(a,b):
        nonlocal body
        assert body.count(a)==1,a;body=body.replace(a,b)
    swap('def da_build_small(ns,early,gate=False,local_reduce=False):','def da_build_small(ns,early,gate=False,local_reduce=False,requests=None):')
    swap('                if e!=early:continue','                if e!=early or requests is not None and (t,k,arm) not in requests:continue')
    # Compile each distinct schedule family only when the selected cells use it.
    swap('    small={}\n    for early in (False,True):\n        for gate in (False,True):\n            mod,index,src=da_build_small(ns,early,gate,local_reduce);',
         '    roles={r:w for r,w in roles.items() if r in DA_ROLES};st["roles"]=roles\n    small={}\n    for early in (False,True):\n        for gate in (False,True):\n            requests={(t,k,a) for t,k,a in DA_REQUESTS if DA_SMALL[a][2]==early and (not gate or (t,"gate_apply") in DA_SELECTED)}\n            if not requests:continue\n            mod,index,src=da_build_small(ns,early,gate,local_reduce,requests);')
    swap('    for name,w in roles.items():\n        b=', '    for name,w in roles.items():\n        if not any(t<=32 and r==name for t,r in DA_NORMALIZED):continue\n        b=')
    swap("            if actual!=name or name=='unpooler_slice' and t>8:continue", "            if actual!=name or name=='unpooler_slice' and t>8:continue\n            if (t,name) not in DA_NORMALIZED:continue")
    swap('            for arm,(wk,s,early) in enumerate(DA_SMALL):\n                mod,index=', '            for arm,(wk,s,early) in enumerate(DA_SMALL):\n                if DA_NORMALIZED[t,name]!=arm:continue\n                mod,index=')
    swap("                if name=='gate':", "                if name=='gate' and (t,'gate_apply') in DA_SELECTED:")
    swap('        for arm,(bn,s,pf,early) in enumerate(DA_WIDE):\n            inst=', '        for arm,(bn,s,pf,early) in enumerate(DA_WIDE):\n            if not any(tt==t and aa==arm for (tt,r),aa in DA_NORMALIZED.items()):continue\n            inst=')
    swap('    for early in (False,True):\n        src=da_wide_source', '    for early in (False,True):\n        if not any(t>32 and DA_WIDE[a][3]==early for (t,r),a in DA_NORMALIZED.items()):continue\n        src=da_wide_source')
    swap("        if name=='unpooler_slice':continue", "        if name=='unpooler_slice' or not any(t>32 and r==name for t,r in DA_NORMALIZED):continue")
    swap('        for s in (1,2,4,8):\n            a=', '        for s in (1,2,4,8):\n            if not any(t>32 and r==name and DA_WIDE[arm][1]==s for (t,r),arm in DA_NORMALIZED.items()):continue\n            a=')
    swap('            for arm,(bn,s,pf,early) in enumerate(DA_WIDE):\n                st[', '            for arm,(bn,s,pf,early) in enumerate(DA_WIDE):\n                if DA_NORMALIZED.get((t,name))!=arm:continue\n                st[')
    constants='\nDA_SELECTED='+repr(selection)+'\nDA_NORMALIZED='+repr(normalized)+'\nDA_ROLES='+repr(roles)+'\nDA_REQUESTS='+repr(requests)+'\n'
    return body+constants

INTEGRATE='''
_da_layer_prepare=prepare
_da_layer_project=_project_variant
_da_layer_best7_project=_best7_project
_da_layer_gate_apply=_gate_split_apply
DA_LOCAL_REDUCE=False
DA_WIDE_STANDIN=False

def prepare(weights):
    state=_da_layer_prepare(weights)
    state['da']=da_prepare(weights,DA_LOCAL_REDUCE,DA_WIDE_STANDIN)
    state['da_prepare_s']=dict(state['da']['times'])
    return state

def _project_variant(x,w,name,state,variant,out_dtype=torch.bfloat16):
    key=(x.shape[0],'unpooler' if name=='unpooler_slice' else name)
    if key in DA_SELECTED and 'da' in state:return da_project(x,key[1],state['da'],DA_SELECTED[key])
    return _da_layer_project(x,w,name,state,variant,out_dtype)

def _best7_project(x,name,state,plan,out_dtype=torch.bfloat16):
    key=(x.shape[0],'unpooler' if name=='unpooler_slice' else name)
    if key in DA_SELECTED and 'da' in state:return da_project(x,key[1],state['da'],DA_SELECTED[key])
    return _da_layer_best7_project(x,name,state,plan,out_dtype)

def _gate_split_apply(x,y,state):
    key=(x.shape[0],'gate_apply')
    if key in DA_SELECTED and 'da' in state:return da_project(x,'gate',state['da'],DA_SELECTED[key],y)
    return _da_layer_gate_apply(x,y,state)
'''
def patch(source,selection):
    assert 'DA_SELECTED=' not in source
    aliases='\nimport torch,hashlib\n_nf_vendor=globals()["_nf_vendor"]\n_nf_group_pack=globals()["_nf_group_pack"]\n_project_variant=globals()["_project_variant"]\n_best7_project=globals()["_best7_project"]\n_gate_split_apply=globals()["_gate_split_apply"]\n'
    result=source+aliases+helper(selection)+'\n'+INTEGRATE
    compile(result,'<P10 dense selected>','exec');return result
if __name__=='__main__':
    assert len(sys.argv)==4
    selection={(t,r):a for t,r,a in json.loads(Path(sys.argv[3]).read_text())}
    source=patch(Path(sys.argv[1]).read_text(),selection);Path(sys.argv[2]).write_text(source)
    print(hashlib.sha256(source.encode()).hexdigest())
