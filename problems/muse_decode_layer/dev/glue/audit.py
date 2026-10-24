"""Pin compute and packed-weight provenance, plus the exact timed launch chain."""
from pathlib import Path
import ast,hashlib,json,sys,importlib.util,time
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best5.py';CAND=BASE.with_name('cand_r3_best5_glue.py')
a=ast.parse(BASE.read_text());b=ast.parse(CAND.read_text())
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150'
def funcs(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
def literals(t):return {k.id:ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) for k in n.targets if isinstance(k,ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
fa,fb=funcs(a),funcs(b);la,lb=literals(a),literals(b)
for k in la:assert la[k]==lb[k],k
unchanged=['_gemv_kernel','gemv','_project_variant','project','_dense_project','_tune_projections','_tune_measured_lt',
'_seed_v9_gemv_choices','_v9_module','v9_prepare','_repack','_prepare_weights','experts_v9','v9_kernel_fn',
'_best5_nt2_module','_best5_tune_moe','_best5_fp32','_best5_fixture','stage_attention','_layer_stage_attention',
'_layer_attention_custom','_attn_port_variant','_tune_attention_portfolio','_norm_gate_kernel',
'_attn_split_kernel','_attn_combine_kernel','attention_triton','attention_fa3','_attention_base']
for k in unchanged:assert ast.dump(fa[k],include_attributes=False)==ast.dump(fb[k],include_attributes=False),k
body=ast.get_source_segment(CAND.read_text(),fb['kernel_fn'])
assert 'torch.zeros' not in body and 'sync' not in body and 'o_and_pool' not in body
assert body.count('boundary(')==2 and body.count('route_and_norm(')==1
assert body.count('project(')==4 and body.count('experts_v9(')==1 and body.count('stage_attention(')==1
assert '_group_barrier' not in fb and '_boundary_kernel' not in fb and '_route_norm_kernel' not in fb
for name in ('_boundary_fast','_route_fast'):
    node=fb[name]
    assert not any(isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='torch' for n in ast.walk(node))
row=dict(sha256=hashlib.sha256(CAND.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
         unchanged_native_literals=list(la),unchanged_functions=unchanged,
         timed_launch_chain=['QKV','attention','O','boundary1','pool_router','route_norm','V9 NT1/NT2','unpooler','boundary2'],
         glue_launches=3,glue_clears=0,global_glue_atomics=0,retained_dequantized_weight_banks=0)
if '--deadline' in sys.argv:
    spec=importlib.util.spec_from_file_location('glue_deadline_audit',CAND);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod._BEST5_DEADLINE=time.monotonic()-1
    def forbidden(*a,**kw):raise AssertionError('expired glue tuner attempted GPU work')
    original=mod.torch.randn
    try:
        mod.torch.randn=forbidden
        state={};mod._tune_glue(state,None,None)
        assert state['glue_geometry']==[]
    finally:mod.torch.randn=original
    row['expired_glue_tuner_checked']=True
(HERE/'audit.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row,indent=2))
