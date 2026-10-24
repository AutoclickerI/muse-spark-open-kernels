"""Prepare the one-LOP3 NT1 cell independently of lab controls and diagnostics."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    tree=ast.parse((ROOT/'prof/lab/moe_m128_issue_limit.py').read_text())
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    call=functions['issue_nt_call'];call.args.args=call.args.args[:2];call.args.defaults=[]
    class Specialize(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='prefix':return ast.copy_location(ast.Constant('nlop'),n)
            return n
        def visit_If(self,n):
            if any(isinstance(v,ast.Name) and v.id=='trace' for v in ast.walk(n.test)):return []
            return self.generic_visit(n)
    call=Specialize().visit(call)
    module=functions['issue_module']
    module.body[-2]=ast.parse("ISSUE_OPTS['nlop']=mod.nlop_init()").body[0]
    for n in tree.body:
        if isinstance(n,ast.Assign):
            name=getattr(n.targets[0],'id','')
            if name=='ISSUE_CUDA':n.value=ast.Constant((HERE/'nlop.cu').read_text())
            if name=='ISSUE_CPP':n.value=ast.Constant('#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t nlop_init();int64_t nlop_smem(int64_t,int64_t);int64_t nlop_resident(int64_t,int64_t);void nlop_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);')
            if name=='ISSUE_FUNCTIONS':n.value=ast.parse(repr(['nlop_'+f for f in ('init','smem','resident','run')]),mode='eval').body
    symbols={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):symbols[n.name]=n
        if isinstance(n,ast.Assign):
            for t in n.targets:
                for target in ast.walk(t):
                    if isinstance(target,ast.Name) and isinstance(target.ctx,ast.Store):symbols[target.id]=n
    selected=set();pending=['issue_nt_call','issue_nt_prepare']
    while pending:
        n=symbols.get(pending.pop())
        if n is None or id(n) in selected:continue
        selected.add(id(n));pending.extend(v.id for v in ast.walk(n) if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Load))
    tree.body=[n for n in tree.body if id(n) in selected]
    names={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):names[n.name]='m128_nt1lop_'+n.name.lstrip('_')
        if isinstance(n,ast.Assign):
            for t in n.targets:
                for target in ast.walk(t):
                    if isinstance(target,ast.Name) and isinstance(target.ctx,ast.Store):names[target.id]='M128_NT1LOP_'+target.id.lstrip('_')
    class Rename(ast.NodeTransformer):
        def visit_FunctionDef(self,n):n.name=names.get(n.name,n.name);return self.generic_visit(n)
        def visit_Name(self,n):n.id=names.get(n.id,n.id);return n
    tree=ast.fix_missing_locations(Rename().visit(tree))
    text='"""M128 NT1 exact one-LOP3 blend; enable only after its paired H200 receipt wins."""\nimport functools\nimport hashlib\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'+ast.unparse(tree)+'\n'
    assert 'norig' not in text and 'ntrace' not in text and 'idist' not in text
    out=HERE/'dropin_nt1_lop.py';out.write_text(text)
    manifest=dict(path=str(out.relative_to(ROOT)),sha256=hashlib.sha256(text.encode()).hexdigest(),ready=False,
        prepare='m128_nt1lop_issue_nt_prepare(weights)',call='m128_nt1lop_issue_nt_call(data,bank)',
        best7_rows=['(16,8,2048)','(16,8,0)'],reason='H200 paired seed-mean win and final full-layer gate required')
    (HERE/'dropin_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':main()
