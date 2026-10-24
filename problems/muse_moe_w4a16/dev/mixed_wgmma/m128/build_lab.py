"""Freeze the shared producer, M128 scheduler and independent public-router gates."""
import ast
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from build_prefill_lab import subset
from source import source,CONFIGS

ROOT=HERE.parents[4]


def main():
    control=(ROOT/'prof/lab/moe_pipeline_seed.py').read_text();tree=ast.parse(control)
    # Keep V9 and masked Humming as controls; raw17 is unrelated here.
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name=='pp_control_prepare':
            n.body=[x for x in n.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Name) and x.test.id=='raw')]
        if isinstance(n,ast.FunctionDef) and n.name=='pp_reference':
            n.body=[x for x in n.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Compare))]
    ctrl=subset(ast.unparse(tree),('dc_weights','dc_fixtures','dc_oracles','pp_control_prepare','pp_reference','MWGraph','rv_kernel_fn'))
    engine=subset((HERE.parent/'engine.py').read_text(),('mw_repack_planes',))
    fused=ast.parse((HERE.parent/'fused.py').read_text())
    helpers='\n'.join(ast.unparse(n) for n in fused.body if isinstance(n,ast.FunctionDef) and n.name in ('ft_prepare','ft_from_packed'))
    helpers=helpers.replace('mod=ft_module()', 'mod=ABI()')
    adapter=ast.parse((HERE/'adapter.py').read_text())
    adapter.body=[n for n in adapter.body if not (isinstance(n,ast.ImportFrom) and n.module in ('source','fused'))]
    for n in adapter.body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='M128_CUDA' for t in n.targets):n.value=ast.Constant(source())
        if isinstance(n,ast.FunctionDef) and n.name=='prepare':n.name='m128_prepare'
        if isinstance(n,ast.FunctionDef) and n.name=='work':n.name='m128_work'
        if isinstance(n,ast.FunctionDef) and n.name=='call':
            n.name='m128_call'
            for node in ast.walk(n):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='work':node.func.id='m128_work'
    code=ast.unparse(ast.fix_missing_locations(adapter))
    out=ROOT/'prof/lab/moe_m128_one_read.py'
    out.write_text(ctrl+'\n'+engine+'\nCONFIGS='+repr(CONFIGS)+'\n'+helpers+'\n'+code+'\n'+(HERE/'driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
