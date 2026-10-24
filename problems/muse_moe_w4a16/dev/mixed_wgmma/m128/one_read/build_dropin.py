"""Emit one fixed M128 cell from a selected expert-wide recipe, without lab controls."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--backend',choices=('or_rs','or_ss64','or_ss128'),required=True)
    ap.add_argument('--config',type=int);ap.add_argument('--grid',type=int,required=True);args=ap.parse_args()
    config=args.config if args.config is not None else (3 if args.backend=='or_rs' else 0)
    assert config in ((3,4,5) if args.backend=='or_rs' else (0,))
    source=(ROOT/'prof/lab/moe_m128_no_reread.py').read_text();tree=ast.parse(source)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    plan=functions['or_plan'];cap=next(i for i,n in enumerate(plan.body) if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='cap' for t in n.targets))
    header=next(i for i,n in enumerate(plan.body) if isinstance(n,ast.If))
    plan.body=plan.body[:header]+ast.parse(f"assert m==128, 'expert-wide cell selected only for M128'\nnt=128\nwhich={config}\ngrid={args.grid}\nprefix={args.backend!r}").body+plan.body[cap:]
    plan.args.args=plan.args.args[:2];plan.args.defaults=[]
    call=functions['or_call'];call.args.args=call.args.args[:2];call.args.defaults=[]
    for n in ast.walk(call):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='or_plan':n.args=n.args[:2]
    module=functions['or_module'];module.body[-2]=ast.parse(f"getattr(mod,{args.backend+'_init'!r})()").body[0]
    native=(HERE/(args.backend+'.cu')).read_text()
    a=native.index('#define FT_CASES(F)');b=native.index('\n}',a)
    line=next(line.strip() for line in native[a:b].splitlines()[1:] if line.strip().startswith('F('+str(config)+','))
    native=native[:a]+'#define FT_CASES(F) '+line.rstrip(' \\')+native[b:]
    cpp='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\n'
    p=args.backend
    cpp+=f'int64_t {p}_init();std::vector<int64_t> {p}_info(int64_t,int64_t);void {p}_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void {p}_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);void {p}_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);'
    for n in tree.body:
        if isinstance(n,ast.Assign):
            name=getattr(n.targets[0],'id','')
            if name=='OR_CUDA':n.value=ast.Constant(native)
            if name=='OR_CPP':n.value=ast.Constant(cpp)
            if name=='OR_FUNCTIONS':n.value=ast.parse(repr([p+'_'+f for f in ('init','info','route','run','finish')]),mode='eval').body
    symbols={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):symbols[n.name]=n
        if isinstance(n,ast.Assign):
            for t in n.targets:
                for v in ast.walk(t):
                    if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Store):symbols[v.id]=n
    selected=set();pending=['or_call','issue_prepare']
    while pending:
        n=symbols.get(pending.pop())
        if n is None or id(n) in selected:continue
        selected.add(id(n));pending.extend(v.id for v in ast.walk(n) if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Load))
    tree.body=[n for n in tree.body if id(n) in selected];names={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names[n.name]='m128_one_'+n.name.lstrip('_')
        if isinstance(n,ast.Assign):
            for t in n.targets:
                for v in ast.walk(t):
                    if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Store):names[v.id]='M128_ONE_'+v.id.lstrip('_')
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        def visit_Name(self,n):n.id=names.get(n.id,n.id);return n
    tree=ast.fix_missing_locations(Rename().visit(tree))
    text='"""Fixed M128 expert-wide recoded MoE cell; H200 receipt controls promotion."""\nimport functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'+ast.unparse(tree)+'\n'
    text+='\ndef m128_one_prepare(weights):\n    m128_one_or_module()\n    return m128_one_issue_prepare(weights)\n'
    text+='\ndef m128_one_experts(x,scores,local_ids,state):\n    data=dict(hidden_states=x,topk_ids=local_ids,topk_weights=scores,config=dict(local_start=0))\n    return m128_one_or_call(data,state)\n'
    out=HERE/'dropin.py';out.write_text(text)
    manifest=dict(path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),backend=args.backend,config=config,grid=args.grid,
        ready=False,reason='Provisional locally gated cell. Enable only after a complete paired H200 win and full-layer gate.',
        prepare='m128_one_prepare(weights)',call="m128_one_experts(x,scores,local_ids,state['moe_one_bank'])",best7_rows=['(16,8,2048)','(16,8,0)'])
    (HERE/'dropin_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':main()
