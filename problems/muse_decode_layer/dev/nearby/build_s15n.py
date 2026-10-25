"""Replace S15's exact dispatch with supported ranges; retain every CUDA byte."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
from cases import S15_SHA
H=Path(__file__).resolve().parent;P=H.parents[1]


def main():
    original=(P/'cand_r3_best7_s15.py').read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==S15_SHA
    text=original
    def one(old,new):
        nonlocal text
        assert text.count(old)==1,(old[:100],text.count(old));text=text.replace(old,new)
    one("plan=state['best7_plans'].get(key,state['best7_plans']['other'])", "plan=_s15n_plan(cfg,state)")
    one("SMALLM_FLAGS = {1: 95, 8: 69, 16: 69, 32: 69}","SMALLM_FLAGS = {1: 95, 64: 69}")
    one("SMALLM_FLAGS[data['hidden_states'].shape[0]]", "SMALLM_FLAGS[1 if data['hidden_states'].shape[0]==1 else 64]")
    one("(cfg['batch'], bool(cfg['window'])) not in GATE_SPLIT_ROWS", "(_s15n_nearest(cfg['batch'],(1,16,32)), bool(cfg['window'])) not in GATE_SPLIT_ROWS")
    one("if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state.get('cluster_rows',()):", "key=(_s15n_nearest(cfg['batch'],(1,16,32)),cfg['q_len'],bool(cfg['window']))\n    if key in state.get('cluster_rows',()):")
    one("if (cfg['batch'],cfg['q_len'],bool(cfg['window'])) in state['partial_port_rows']:", "if key in state['partial_port_rows']:")
    one("vq8=state.get('vq8_rows',{}).get((cfg['batch'],cfg['q_len'],bool(cfg['window'])))", "vq8=state.get('vq8_rows',{}).get((_s15n_nearest(cfg['batch'],(16,32)),8,False)) if cfg['batch']>1 and cfg['q_len']==8 and not cfg['window'] else None")
    one("choice=state['attention_plan'].get((cfg['batch'],cfg['q_len'],bool(cfg['window'])),('layer',None))", "choice=state['attention_plan'].get(key,('layer',None))")
    one("cfg = CD_ROWS.get((x.shape[0], name))", "cfg = CD_ROWS.get((_s15n_nearest(x.shape[0],_TUNE_TOKENS), name))")
    one("choice = _GEMV_CHOICES.get((T, N, K))", "keys=[m for m,n,k in _GEMV_CHOICES if (n,k)==(N,K)]\n    choice = _GEMV_CHOICES.get((_s15n_nearest(T,keys),N,K)) if keys else None")
    one("if x.shape[0] in (32,) and tuple(w.shape)==(4608,8192):", "if _s15n_nearest(x.shape[0],(1,8,16,32))==32 and tuple(w.shape)==(4608,8192):")
    old="""    variant = state.get("proj_plan", {}).get((x.shape[0], name), "custom")
    return _project_variant(x, w, name, state, variant, out_dtype=out_dtype)"""
    new="""    t=x.shape[0];anchor,variant=_s15n_projection(t,name,state)
    if isinstance(variant,tuple) and variant[0] in ('lt','short_lt') and t!=anchor:
        if t>anchor:
            rows=[_materialize(project(x[i:i+anchor],w,name,state,out_dtype)).to(out_dtype)
                  for i in range(0,t,anchor)]
            return torch.cat(rows,dim=0).unsqueeze(0),1
        padded=F.pad(x,(0,0,0,anchor-t))
        p,s=_project_variant(padded,w,name,state,variant,out_dtype=out_dtype)
        return p[:,:t],s
    return _project_variant(x,w,name,state,variant,out_dtype=out_dtype)"""
    one(old,new)
    # All selected small NF sources have CM=0; MT bounds are checked by the
    # range selector. Wide O/MW kernels explicitly mask m and tile ceil(m/BN).
    one("key=x.shape[0],name;p=state['plans'].get(key)","key=_s15n_nf_key(x.shape[0],name);p=state['plans'].get(key)")
    one("p=state['plans'].get((x.shape[0],name))", "p=state['plans'].get(_s15n_nf_key(x.shape[0],name))")
    old="""    key=(x.shape[0],name)
    if 'nf_port' in state and (key in NF_PORT_SELECTED or NF_PORT_FORCE and key in NF_SMALL.keys()|NF_WIDE):"""
    new="""    key=_s15n_nf_key(x.shape[0],name)
    if 'nf_port' in state and key is not None:"""
    assert text.count(old)==2;text=text.replace(old,new)
    text+='\n\n'+(H/'ranges.py.inc').read_text()
    target=P/'cand_r3_best7_s15n.py';target.write_text(text)
    patch=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),
        fromfile='a/problems/muse_decode_layer/cand_r3_best7_s15.py',tofile='b/problems/muse_decode_layer/cand_r3_best7_s15n.py'))
    (H/'s15n.patch').write_text(patch)
    a,b=ast.parse(original),ast.parse(text)
    strings=lambda tree:{n.targets[0].id:n.value.value for n in tree.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)
        and isinstance(n.value.value,str)}
    assert strings(a)==strings(b),'CUDA/vendor/literal sources changed'
    meta=dict(base_sha256=S15_SHA,candidate_sha256=hashlib.sha256(text.encode()).hexdigest(),
              changed_lines=sum(l.startswith(('+','-')) and not l.startswith(('+++','---')) for l in patch.splitlines()),
              patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),literal_sources_identical=len(strings(a)))
    (H/'s15n_identity.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)


if __name__=='__main__':main()
