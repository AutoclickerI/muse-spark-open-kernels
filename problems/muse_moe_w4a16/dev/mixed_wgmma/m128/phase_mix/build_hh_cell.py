"""Freeze the admitted H_H recipe for best7 without lab controls or a portfolio."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    wrapper=next((ROOT/'runs/muse_moe_w4a16').glob('*770809cf159f41e2.py'))
    tree=ast.parse(wrapper.read_text())
    payloads=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')
    _,sha,text=next(v for v in payloads if v[0]=='moe_m128_phase_mix')
    assert hashlib.sha256(text.encode()).hexdigest()=='f3c2796944c60b77aa5d63fd61e766a386bcdcfaf526b8ed01ab19e5367e5736'
    assert hashlib.sha256(text.encode()).hexdigest().startswith(sha)
    sha=hashlib.sha256(text.encode()).hexdigest()
    tree=ast.parse(text);fn={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    # Assert the complete embedded source, then retain exact admitted kernels.
    cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='PM_CUDA')
    codes={p:(HERE/(p+'.cu')).read_text() for p in ('idist','norig','or_rs','pm_nt','pm_meta')}
    assert '\n'.join(codes.values())==cuda
    selected=('idist','pm_nt','pm_meta')
    # Scales stay packed uint8 codes; the BF16 word forms in registers (S7, commit 69da04e).
    import sys;sys.path.insert(0,str(ROOT/'problems/muse_decode_layer/dev/s7'));from build import CUDA_EDITS
    for old,new,count in CUDA_EDITS:
        assert codes['idist'].count(old)==count,old;codes['idist']=codes['idist'].replace(old,new)
    cpp=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='PM_CPP')
    cpp='\n'.join(l for l in cpp.splitlines() if l.startswith('#include') or l.startswith(('int64_t idist_','int64_t pm_nt_')))+'\n'
    funs=['idist_'+f for f in ('init','info','route','run','finish')]+['pm_nt_'+f for f in ('init','smem','resident','run')]+['pm_meta']
    module=fn['pm_module']
    for n in ast.walk(module):
        if isinstance(n,ast.Constant) and n.value==('idist','or_rs','norig','pm_nt'):n.value=('idist','pm_nt')
        if isinstance(n,ast.Tuple) and ast.literal_eval(n)==('idist','or_rs','norig','pm_nt'):n.elts=[ast.Constant('idist'),ast.Constant('pm_nt')]
    plan=fn['pm_plan'];plan.args.args=[plan.args.args[i] for i in (0,1,3)];plan.args.defaults=[]
    for n in ast.walk(plan):
        if isinstance(n,ast.Name) and n.id=='word':n.id='base'
    body=[]
    for n in plan.body:
        if isinstance(n,ast.Assign):
            names=[getattr(t,'id','') for t in n.targets]
            if 'rs13' in names or 'rs2' in names:continue
            if 'vectors' in names:n.value=ast.parse('[word13,word2,big13,big2,nv]',mode='eval').body
        if isinstance(n,ast.FunctionDef) and n.name=='phase':
            n.args.args=n.args.args[1:]
            n.body=ast.parse('''_=owners
small=1;grid=native['grid']*mod.pm_nt_resident(stages,m,which,small)
mod.pm_nt_run(ptrs[4],which,small,m,stages,grid,stream())
mod.idist_run(ptrs[2 if which else 3],which,1,1056,start,1,0,m,cap,stream())
''').body
        if isinstance(n,ast.FunctionDef) and n.name in ('w13','w2'):
            n.body=ast.parse('phase('+('1' if n.name=='w13' else '0')+')').body
        body.append(n)
    plan.body=body
    call=fn['pm_call'];call.args.args=[call.args.args[i] for i in (0,1,3)];call.args.defaults=[]
    for n in ast.walk(call):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='pm_plan':n.args=[n.args[i] for i in (0,1,3)]
    for n in tree.body:
        if isinstance(n,ast.Assign):
            key=getattr(n.targets[0],'id','')
            if key=='PM_CUDA':n.value=ast.Constant('\n'.join(codes[p] for p in selected))
            if key=='PM_CPP':n.value=ast.Constant(cpp)
            if key=='PM_FUNCTIONS':n.value=ast.parse(repr(funs),mode='eval').body
    tree=ast.fix_missing_locations(tree)
    bindings={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):bindings[n.name]=n
        if isinstance(n,ast.Assign):
            for target in n.targets:
                for v in ast.walk(target):
                    if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Store):bindings[v.id]=n
    selected_nodes=set();pending=['issue_prepare','issue_nt_prepare','pm_call']
    while pending:
        n=bindings.get(pending.pop())
        if n is None or id(n) in selected_nodes:continue
        selected_nodes.add(id(n));pending.extend(v.id for v in ast.walk(n) if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Load))
    tree.body=[n for n in tree.body if id(n) in selected_nodes]
    names={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names[n.name]='m128_hh_'+n.name.lstrip('_')
        if isinstance(n,ast.Assign):
            for t in n.targets:
                for v in ast.walk(t):
                    if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Store):names[v.id]='M128_HH_'+v.id.lstrip('_')
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        def visit_Name(self,n):n.id=names.get(n.id,n.id);return n
    tree=ast.fix_missing_locations(Rename().visit(tree))
    text='"""H_H M128 (H200 770809cf recipe): NT1<=16, WGMMA32>16 in both phases; packed uint8 scale codes (S7 kernel)."""\nimport functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'+ast.unparse(tree)+'\n'
    text+='''
def hh_prepare(weights):
    w=dict(weights,post_norm_w=weights.get('post_expert_norm',weights.get('post_norm_w')))
    base=m128_hh_issue_prepare(w)
    return dict(base=base,native=m128_hh_issue_nt_prepare(w))

def hh_experts(x,scores,local_ids,state,local_start=0):
    assert x.shape[0]==128, 'H_H cell is selected only for M128'
    data=dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=local_start))
    return m128_hh_pm_call(data,state['base'],state['native'])
'''
    assert 'or_rs' not in text and 'norig' not in text and 'word_bank' not in text and 'torch.int16' not in text
    out=HERE/'hh_cell.py';out.write_text(text)
    meta=dict(path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),ready=True,receipt='770809cf159f41e2',
        prepare='hh_prepare(weights)',call='hh_experts(x,scores,local_ids,state,local_start=0)',tokens=128,
        backend='both W13 and W2: recoded one-LOP3 NT1 for count<=16; uint8-scale WGMMA32 config1/grid1056 for count>16 (S7 kernel, BF16 word in registers); distributed finalizer',
        mean_us=234.44,D32g_us=272.70,mean_paired_ratio=.8629,fixtures=36,seed_count=12,
        source_payload_sha256=sha,embedded_sources={p:hashlib.sha256(codes[p].encode()).hexdigest() for p in selected},
        note='Fixed specialization of accepted launch/packing/bodies; actual public wrapper local gate required; whole-layer H200 gain unverified.')
    (HERE/'hh_handoff.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['path'],meta['sha256'],len(text))


if __name__=='__main__':main()
