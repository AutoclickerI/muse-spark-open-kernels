"""AST inventory of S15's dispatch tables, shape branches and Python assertions."""
import ast
import hashlib
import json
from pathlib import Path
from cases import S15_SHA, cases
H=Path(__file__).resolve().parent;R=H.parents[3]
p=R/'problems/muse_decode_layer/cand_r3_best7_s15.py';s=p.read_text()
assert hashlib.sha256(s.encode()).hexdigest()==S15_SHA
t=ast.parse(s)
shape_words=('shape','batch','tokens','q_len','window','m','M','t','T','length','ns','splits')
functions={'kernel_fn','_best7_run','_best7_attention','_gate_split_on','_best7_project',
 '_best7_expert','_best6_experts','_best6_call','experts_v9','v9_kernel_fn','smallm_experts',
 '_project_variant','project','gemv','_gb_gemv','_dense_project','_best7_boundary',
 '_best7_route','_layer_stage_attention','_attn_port_variant','_vq8_attention',
 'cs_attention','cs_plan','nf_project','_best7_prepare_settings','_s4_gate_moe'}
tables={'BEST7_SETTINGS','GLUE_CL_ROWS','BEST7_DENSE_CHAIN','SMALLM_FLAGS','SMALLM_FLAGS_BUILT',
 '_TUNE_TOKENS','GB_ROWS','GATE_SPLIT_ROWS','GATE_SPLIT_CFG','GATE_SPLIT_QKV3','GEMV_CFG',
 'GEMV_MAX_T','SLICED_MAX_TOKENS','CD_ROWS','PROJ_PINS','VDSK_SHORTLIST','_MEASURED_LT',
 '_VQ8_ROWS','_PP_ROWS','_CLUSTER_ROWS','_CLUSTER_NK','NF_PORT_SELECTED','NF_SMALL','NF_WIDE','NF_PORT_FORCE'}
result=dict(candidate=str(p.relative_to(R)),sha256=S15_SHA,tables=[],branches=[],assertions=[],cases=cases())
for n in t.body:
 if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in tables for z in n.targets):
  result['tables'].append(dict(line=n.lineno,source=ast.unparse(n)))
 if isinstance(n,ast.FunctionDef) and n.name in functions:
  for z in ast.walk(n):
   if isinstance(z,(ast.If,ast.IfExp)):
    cond=ast.unparse(z.test)
    if any(w in cond for w in shape_words):result['branches'].append(dict(function=n.name,line=z.lineno,condition=cond))
for n in ast.walk(t):
 if isinstance(n,ast.Assert):result['assertions'].append(dict(line=n.lineno,condition=ast.unparse(n.test)))
(H/'branch_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(len(result['tables']),'tables',len(result['branches']),'branches',len(result['assertions']),'assertions')
