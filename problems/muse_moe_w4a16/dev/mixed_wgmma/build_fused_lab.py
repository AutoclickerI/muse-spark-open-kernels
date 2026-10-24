"""Pack corrected-vector, pipeline and fused M256 variants with matched controls."""
import ast
import hashlib
from pathlib import Path
from build_prefill_lab import subset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def inline(name,var,filename):
    text=(HERE/filename).read_text()
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module=='engine')]
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id==var for x in n.targets):
            n.value=ast.Constant((HERE/(name+'.cu')).read_text())
        if isinstance(n,ast.FunctionDef) and n.name in ('vec_module','pipe_module','ft_module'):
            n.decorator_list=[]
            n.body=ast.parse('return fused_lab_module()').body
    return ast.unparse(ast.fix_missing_locations(tree))


def main():
    # Reuse the independently gated public-router controls; remove unused raw17.
    source=(ROOT/'prof/lab/moe_pipeline_seed.py').read_text();tree=ast.parse(source)
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and n.name=='pp_control_prepare':
            n.body=[x for x in n.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Name) and x.test.id=='raw')]
        if isinstance(n,ast.FunctionDef) and n.name=='pp_reference':
            n.body=[x for x in n.body if not (isinstance(x,ast.If) and isinstance(x.test,ast.Compare))]
    control=subset(ast.unparse(tree),('dc_weights','dc_fixtures','dc_oracles','pp_control_prepare','pp_reference','MWGraph','rv_kernel_fn'))
    engine=subset((HERE/'engine.py').read_text(),('mw_repack_planes','mw_route','mw_prefix','mw_swiglu','mw_norm','mw_combine'))
    out=ROOT/'prof/lab/moe_fused_m256.py'
    native='\n'.join((HERE/(n+'.cu')).read_text() for n in ('vector','pipeline','fused'))
    module='''
@functools.lru_cache(None)
def fused_lab_module():
    target='-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability()==(9,0) else '-gencode=arch=compute_120,code=sm_120'
    mod=load_inline(name='fused_lab_'+hashlib.sha256(COMBINED_CUDA.encode()).hexdigest()[:12],
        cpp_sources=VECTOR_CPP+PIPE_CPP+FT_CPP,cuda_sources=COMBINED_CUDA,
        functions=['vec_run','pipe_init','pipe_info','pipe_run','ft_init','ft_info','ft_route','ft_run'],
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',target],verbose=False)
    mod.pipe_init();mod.ft_init();return mod
'''
    out.write_text(control+'\n'+engine+'\n'+inline('vector','VECTOR_CUDA','vector.py')+'\n'+
        inline('pipeline','PIPE_CUDA','pipeline.py')+'\n'+inline('fused','FT_CUDA','fused.py')+'\nCOMBINED_CUDA='+repr(native)+'\n'+module+'\n'+(HERE/'fused_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
