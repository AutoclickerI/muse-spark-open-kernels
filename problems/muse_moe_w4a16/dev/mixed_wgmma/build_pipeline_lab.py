"""One payload: deeper-pipeline W13 ablations, R0 raw17, and 12-seed E2E controls."""
import ast
import hashlib
from pathlib import Path
from build_prefill_lab import subset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    path=ROOT/'prof/lab/moe_decode_adapters.py'
    controls=subset(path.read_text(),('rv_prepare','rv_kernel_fn','dc_weights','dc_fixtures','dc_oracles',
        'pf__grouped_module','pf__raw_repack','pf__encode_blob_fast','pf__raw_experts_tiled',
        'pf__Experts','pf__experts_humming_masked','pf__portfolio_swiglu','MWGraph'))
    tree=ast.parse(controls)
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pf_GROUPED_SRC' for t in n.targets):
            src=ast.literal_eval(n.value)
            # The measured raw17 recipe alone; raw64 is an unrelated open experiment.
            second=src.find('#include <torch/extension.h>',src.index('namespace raw17 {'))
            assert second>0
            n.value=ast.Constant(src[:second])
        if isinstance(n,ast.FunctionDef) and n.name=='pf__grouped_module':
            n.body=ast.parse('''
def pf__grouped_module():
    global pf__mod
    if pf__mod is None:
        from torch.utils.cpp_extension import load_inline
        pf__mod=load_inline(name='pipe_raw17_'+pf_hashlib.sha256(pf_GROUPED_SRC.encode()).hexdigest()[:12],
            cpp_sources='void raw17_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);',
            cuda_sources=pf_GROUPED_SRC,functions=['raw17_gemm'],extra_cuda_cflags=['-O3','-std=c++20',
              '-gencode=arch=compute_90a,code=sm_90a','-gencode=arch=compute_120,code=sm_120'],verbose=False)
    return pf__mod
''').body[0].body
    controls=ast.unparse(ast.fix_missing_locations(tree))
    engine=subset((HERE/'engine.py').read_text(),('mw_repack_planes','mw_route','mw_prefix','mw_swiglu','mw_norm','mw_combine'))
    pipe=(HERE/'pipeline.py').read_text().replace('from engine import mw_repack_planes,mw_route,mw_prefix,mw_swiglu,mw_norm,mw_combine','')
    pipe=pipe.replace("PIPE_CUDA=Path(__file__).with_name('pipeline.cu').read_text()",'PIPE_CUDA='+repr((HERE/'pipeline.cu').read_text()))
    out=ROOT/'prof/lab/moe_pipeline_seed.py'
    out.write_text('# Control source SHA256 '+hashlib.sha256(path.read_bytes()).hexdigest()+'\n'+controls+'\n'+engine+'\n'+pipe+'\n'+(HERE/'pipeline_driver.py.inc').read_text())
    print(out.relative_to(ROOT),hashlib.sha256(out.read_bytes()).hexdigest(),out.stat().st_size)


if __name__=='__main__':main()
