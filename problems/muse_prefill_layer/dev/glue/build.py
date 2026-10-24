"""Build a standalone full-layer glue portfolio; queued sources are never edited."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent


class ReadyNorm(ast.NodeTransformer):
    def __init__(self,names):self.names=names
    def visit_Name(self,n):
        if n.id in self.names:n.id='_glue_ready'+n.id
        if n.id=='_gather_norm_kernel':n.id='_glue_gather_ready'
        return n
    def visit_FunctionDef(self,n):
        n.name='_glue_ready'+n.name
        return self.generic_visit(n)
    def visit_Expr(self,n):
        v=n.value
        if (isinstance(v,ast.Call) and isinstance(v.func,ast.Subscript)
                and isinstance(v.func.value,ast.Name) and v.func.value.id=='_pre_expert_norm_kernel'):
            return ast.copy_location(ast.Assign(targets=[ast.Name(id=v.args[2].id,ctx=ast.Store())],value=v.args[0]),n)
        return self.generic_visit(n)


def build(output):
    source=(ROOT/'problems/muse_prefill_layer/cand_score_portfolio_routing.py').read_text()
    tree=ast.parse(source)
    nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    clone_names={'_experts_selected','_score_expert','_run_canonical_dg','_experts_humming_masked',
        '_experts_humming','_experts_pipeline','_raw_experts_tiled','_raw_experts_dequant',
        '_run_raw_dg','_experts_canonical'}
    clones=[]
    for name in sorted(clone_names):
        n=ast.parse(ast.get_source_segment(source,nodes[name])).body[0]
        n=ReadyNorm(clone_names).visit(n);ast.fix_missing_locations(n)
        clones.append(ast.unparse(n))
    gather=ast.get_source_segment(source,nodes['_gather_norm_kernel'])
    gather=gather.replace('def _gather_norm_kernel','def _glue_gather_ready')
    gather=gather.replace('x = (x * tl.rsqrt(tl.sum(x * x, 0) / D + 1e-5) * tl.load(w_ptr + c)).to(tl.bfloat16)',
                          'x = x.to(tl.bfloat16)')
    clones.insert(0,'@triton.jit\n'+gather)
    # The parent already measured every dense/expert/attention family on H200.
    # Keep its full preparation and proven choices; spend this run's trial budget on glue.
    n=nodes['prepare'];prep=ast.get_source_segment(source,n)
    start=prep.index('    if IS_HOPPER:')
    end=prep.index('    torch.cuda.synchronize()',start)
    prep=prep[:start]+"    if IS_HOPPER:\n        _glue_seed_controls(st)\n"+prep[end:]
    source=source[:n.col_offset+sum(len(x) for x in source.splitlines(keepends=True)[:n.lineno-1])]+prep+source[sum(len(x) for x in source.splitlines(keepends=True)[:n.end_lineno]):]
    receipt=json.loads(next((ROOT/'runs/muse_prefill_layer').glob('*a30fb411b0c74ffb.json')).read_text())
    line=next(l for l in receipt['result']['log_tail'].splitlines() if l.startswith('SCORE_LAST_SUMMARY '))
    summary=json.loads(line.split(' ',1)[1])
    dense={ast.literal_eval(r['shape']):tuple(r['winner']) for r in summary['rows'] if r['stage']=='dense'}
    experts={int(r['shape']):r['winner'] for r in summary['rows'] if r['stage']=='expert'}
    attention={ast.literal_eval(r['shape']):r['winner'] for r in summary['rows'] if r['stage']=='attention'}
    source+='\n\n'+(HERE/'kernels.py').read_text()+'\n\n'+'\n\n'.join(clones)
    source+='\n\nGLUE_H200_DENSE='+repr(dense)+'\nGLUE_H200_EXPERTS='+repr(experts)+'\nGLUE_H200_ATTENTION='+repr(attention)+'\n'
    source+='\n'+(HERE/'tail.py').read_text()+'\n'
    ast.parse(source)
    p=Path(output)
    if p.exists() and p.read_text()!=source and p.name.startswith('cand_'):
        raise SystemExit('Refusing to overwrite frozen candidate; use a new output filename')
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
    return p


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'runs/screen/glue_prefill/work.py'))
    print(build(ap.parse_args().output))
