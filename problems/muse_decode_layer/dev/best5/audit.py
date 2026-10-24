"""Check exact native provenance, unchanged stages and all expired tuner guards."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,time
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best.py'
CAND=BASE.with_name('cand_r3_best5.py')


def value(tree,name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id==name for k in n.targets)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--deadline',action='store_true');args=ap.parse_args()
    a=ast.parse(BASE.read_text());b=ast.parse(CAND.read_text())
    assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2'
    literals=('V9_CUDA_SRC','V9_CPP_SRC','CUDA_SRC','LT_CUDA_SRC','ATTN_PORT_CUDA_SRC','ATTN_X_CUDA','ATTN_G_CUDA','ATTN_P8_CUDA','PG_RAW')
    for name in literals:assert value(a,name)==value(b,name),name
    native=value(b,'V9_NT2_CUDA_SRC')
    lab=value(ast.parse((ROOT/'prof/lab/moe_v9_reuse_v2.py').read_text()),'base_CUDA_SRC')
    def body(src):
        start=src.index('template <int NT>\n__global__');end=src.index('\nsize_t smem_bytes(',start)
        return src[start:end]
    assert body(native)==body(lab)
    assert 'moe_fused<1>' not in native and 'moe_fused<4>' not in native
    unchanged=('kernel_fn','_prepare_weights','v9_prepare','_v9_module','_repack','boundary','route_and_norm',
               'project','_project_variant','_projection_consumer','o_and_pool','stage_attention','_attn_port_variant',
               '_attn_port_candidates','_attention_reference','_attention_agrees','_attention_store_agrees')
    fa={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)}
    fb={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
    for name in unchanged:assert ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False),name
    result=dict(source_sha256=hashlib.sha256(CAND.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
                nt2_kernel_sha256=hashlib.sha256(body(native).encode()).hexdigest(),unchanged_literals=list(literals),unchanged_functions=list(unchanged))
    if args.deadline:
        spec=importlib.util.spec_from_file_location('best5_audit',CAND);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        def no_work(*a,**kw):raise AssertionError('expired tuner attempted GPU allocation or compilation')
        mod._fa3_on_device=lambda d:True;mod._attn_port_hopper=lambda d:True
        mod._BEST5_DEADLINE=time.monotonic()-1.
        mod._lt_module=no_work;mod._attn_port_ext=no_work
        empty=mod.torch.empty;randn=mod.torch.randn
        try:
            mod.torch.empty=no_work;mod.torch.randn=no_work
            state={}
            mod._best5_tune_moe(state,{},None,None)
            assert state['moe_nt_plan']=={32:1,128:1,256:1}
            proj=mod._tune_projections(state,None,None);assert len(proj)==24
            mod._tune_measured_lt(state,None,None)
            attention=mod._tune_attention_portfolio(state,None,None);assert len(attention)==12
            state['pdl_plan']=mod._tune_pdl(state,None,None);assert len(state['pdl_plan'])==12
            prefix=mod._tune_prefix(state,None,None);assert len(prefix)==6
            result['expired_deadline_tuners']=['moe_nt','projections','measured_lt','attention','pdl','prefix']
        finally:
            mod.torch.empty=empty;mod.torch.randn=randn
    (HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
