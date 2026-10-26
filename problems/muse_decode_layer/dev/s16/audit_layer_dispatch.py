"""Assert that independently composed layer replacements still own their calls."""
import ast,hashlib,json,types
from pathlib import Path
import patches
H=Path(__file__).resolve().parent


def main():
    source,meta=patches.compose(['epi_PAIR']+[n for n in patches.registry() if n.startswith('layer_')])
    tree=ast.parse(source);fn=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    rows={};roles={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            if n.targets[0].id=='_VQ8_ROWS':rows=ast.literal_eval(n.value)
            if n.targets[0].id=='LIBRARY_NATIVE_ROWS':roles=ast.literal_eval(n.value)
        if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='_VQ8_ROWS':rows.update(ast.literal_eval(n.value.args[0]))
    root=next(n for n in fn if n.name=='_best7_attention' and '_s15n_nearest' in ast.get_source_segment(source,n))
    nearest=next(n for n in fn if n.name=='_s15n_nearest')
    ns={'_vq8_attention':lambda q,d,v:v};exec(compile(ast.Module(body=[nearest,root],type_ignores=[]),'<dispatch-proof>','exec'),ns)
    for b,w,expected in ((1,False,(4,2)),(32,True,(4,2)),(16,False,(4,8)),(32,False,(4,8))):
        got=ns['_best7_attention'](None,dict(config=dict(batch=b,q_len=8,window=2048 if w else 0)),dict(vq8_rows=rows,cluster_rows=(),partial_port_rows=()),dict(attention=('layer',None)))
        assert got==expected,(b,w,got)
    project=next(n for n in fn if n.name=='_best7_project' and 's16_native_key' in ast.get_source_segment(source,n))
    ns=dict(torch=types.SimpleNamespace(bfloat16='dtype-only'),LIBRARY_NATIVE_ROWS=roles,_vdsk_project=lambda x,w,v,chain:('vdsk',v,chain));exec(compile(ast.Module(body=[project],type_ignores=[]),'<project-proof>','exec'),ns)
    assert set(roles)=={(128,'pool_router'),(128,'unpooler')}
    for role in ('pool_router','unpooler'):
        got=ns['_best7_project'](types.SimpleNamespace(shape=(128,1)),role,{role:None},{})
        assert got==('vdsk',roles[128,role],True),got
    gemm=next(n for n in fn if n.name=='_vdsk_gemm');project_fn=next(n for n in fn if n.name=='_vdsk_project')
    assert ast.unparse(gemm.body[0])=='tl.extra.cuda.gdc_launch_dependents()'
    assert any(isinstance(n,ast.If) and ast.unparse(n.test)=='CHAIN' for n in gemm.body)
    assert 'CHAIN=chain, launch_pdl=True' in ast.get_source_segment(source,project_fn)
    cuda=[n.value.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)
          and any(getattr(t,'id','').startswith('_s16_epi_') and getattr(t,'id','').endswith('CUDA') for t in n.targets)]
    assert len(cuda)==1 and 'if constexpr(PDL) asm volatile("griddepcontrol.launch_dependents;"' in cuda[0]
    target=H/'generated/layer_compound.py';target.write_text(source)
    result=dict(passed=True,source_sha256=meta['sha256'],checks=['v1full and v32slide VQ8 paths active','original full verify rows preserved','both actual-M VDSK roles active','release and CHAIN launch/wait edits coexist','PAIR actual module carries conditional entry release'])
    (H/'layer_dispatch_proof.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
