"""Compare the new producer with the prefill lane's raw17/masked/DG expert paths on proj."""
import ast
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prof/lab'))
from build_moe_real_routing import namespace


def subset(source,roots):
    tree=ast.parse(source);bindings={};imports=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom,ast.Try)):imports.append(node)
        elif isinstance(node,(ast.FunctionDef,ast.ClassDef)):bindings[node.name]=node
        elif isinstance(node,ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n,ast.Name):bindings[n.id]=node
    selected=set();pending=list(roots)
    pending.extend(n.id for imp in imports for n in ast.walk(imp) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
    while pending:
        name=pending.pop();node=bindings.get(name)
        if node is None or id(node) in selected:continue
        selected.add(id(node))
        pending.extend(n.id for n in ast.walk(node) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load))
    body=[n for n in tree.body if n in imports or id(n) in selected or
          (isinstance(n,ast.Assign) and any(isinstance(t,ast.Subscript) for t in n.targets)
           and 'HUMMING_DISABLE_PARALLEL_BUILD' in ast.unparse(n))]
    return ast.unparse(ast.Module(body=body,type_ignores=[]))


def main():
    path=ROOT/'problems/muse_prefill_layer/cand_mid_sweep.py'
    source=path.read_text()
    roots=('_grouped_module','_raw_repack','_encode_blob_fast','_encode_sf_fast','_RawModule',
           'repack_v6','_raw_experts_tiled','_run_canonical_dg','_Experts','_experts_humming_masked')
    control=subset(source,roots)
    # Serving receives normalized x. Remove the duplicated pre-expert norm from
    # all copied producers while preserving their GEMM/epilogue implementations.
    tree=ast.parse(control)
    raw=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_raw_experts_tiled')
    raw.body=[n for n in raw.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
              and isinstance(n.value.func,ast.Subscript) and isinstance(n.value.func.value,ast.Name)
              and n.value.func.value.id=='_pre_expert_norm_kernel')]
    for n in raw.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='xn' for t in n.targets):
            n.value=ast.Name(id='pooled',ctx=ast.Load())
    control=ast.unparse(ast.fix_missing_locations(tree))
    norm="x = (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-05) * tl.load(w_ptr + c)).to(tl.bfloat16)"
    assert norm in control
    control=control.replace(norm,"x = x.to(tl.bfloat16)")
    control=control.replace('xn = rms(pooled[t]) * (1 + w); copy it to each of the token\'s local-route rows.',
                            'Gather already-normalized input into local-route rows.')
    control=control.replace('self.compile([1 << i for i in range(16)])','self.compile([512,1024,4096,16384])')
    control=control.replace('import deep_gemm\n',"deep_gemm = st['_dg']\n")
    control=namespace(control,'pf_')
    # The shared namespacer rewrites Name nodes, while Python parameters retain
    # their spelling. Restore lexical parameter references (notably D constexpr).
    tree=ast.parse(control)
    for node in ast.walk(tree):
        if isinstance(node,ast.FunctionDef):
            params=node.args.posonlyargs+node.args.args+node.args.kwonlyargs
            restore={'pf_'+a.arg:a.arg for a in params}
            for stmt in node.body:
                for n in ast.walk(stmt):
                    if isinstance(n,ast.Name) and n.id in restore:n.id=restore[n.id]
    control=ast.unparse(tree)
    base=(ROOT/'prof/lab/moe_mixed_wgmma_planes.py').read_text()
    base=base[:base.index('\ndef lab(deadline,weights,state=None):')]
    out=ROOT/'prof/lab/moe_mixed_wgmma_prefill.py'
    out.write_text(base+'\n# Prefill control source SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest()+'\n'+control+'\n'+(HERE/'prefill_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
