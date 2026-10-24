"""Verify frozen S5 scope, exact accepted mixed_parallel math and the M1 O cell."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
m=json.loads((H/'manifest.json').read_text());s=(R/m['source']).read_text()
base=(R/'problems/muse_decode_layer/cand_r3_best7_s5.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()==m['sha256']
assert hashlib.sha256(base.encode()).hexdigest()==m['base_sha256']
def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def assignment(text,key):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
                and any(getattr(t,'id','')==key for t in n.targets))
def literal(text,key):return ast.literal_eval(assignment(text,key).value)
wrapper=(R/m['accepted_wrapper']).read_text()
name,short,payload=next(row for row in literal(wrapper,'_LAB_PAYLOADS') if row[0]=='moe_w13_issue_m256')
assert m['payload_sha256'].startswith(short) and hashlib.sha256(payload.encode()).hexdigest()==m['payload_sha256']
a,b,p=funcs(base),funcs(s),funcs(payload)
changed={'prepare','_best6_call','_best7_gate_moe','_best7_prepare_settings','_best7_stream_op'}
removed={'rc_module','rc_prepare','rc_plan','rc_call','_s3_gate_moe'}
for name in a:
    if name in removed:assert name not in b,name
    elif name not in changed:assert dump(a[name])==dump(b[name]),name
for name in (*m['exact_functions'],'mw_repack_planes','rc_masks','rc_repack'):
    assert dump(p[name])==dump(b[name]),name
for key in ('FX_CUDA','FX_CPP','FX_CONFIGS','WI_CPP'):assert literal(s,key)==literal(payload,key),key
for key in ('S5_HH_CUDA','S5_HH_CPP','S5_HH_FUNCTIONS','S4_NT1_CUDA','NT1_CPP',
            'V9_CUDA_SRC','_V9_RECODED_CUDA_SRC','BEST7_DENSE_CHAIN','SP_CUDA','DENSE_CHAIN_CUDA'):
    assert dump(assignment(s,key))==dump(assignment(base,key)),key
old,new=literal(base,'BEST7_SETTINGS'),literal(s,'BEST7_SETTINGS')
for key,row in old.items():
    if key in ((32,8,2048),(32,8,0)):assert new[key]==(*row[:2],'mixed_parallel_m256',row[3])
    elif key in ((1,1,2048),(1,1,0)):assert new[key]==(*row[:3],{'o':('stream',6,1)})
    else:assert new[key]==row,key
projection=(R/m['projection']['path']).read_text()
assert hashlib.sha256(projection.encode()).hexdigest()==m['projection']['sha256']
assert s.endswith(projection+'\n')
for name,node in funcs(projection).items():assert dump(node)==dump(b[name]),name
assert "wi_call(data,state['s6_bank'],6,2112)" in s
assert "state['s6_bank']=wi_prepare(weights)" in s
assert "moe in ('mixed_parallel_m256','nt1_lop3','hh_m128'):raise" in s
assert 's3_bank' not in s and 'RC_CUDA=' not in s
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
result=dict(sha256=m['sha256'],accepted_payload_exact=True,native_source_exact=True,
            exact_functions=m['exact_functions'],projection_cell_exact=True,
            unchanged_base_functions=len(a)-len(changed)-len(removed),changed_base_functions=sorted(changed),
            removed_m256_engine=sorted(removed),changed_moe_rows=[[32,8,2048],[32,8,0]],
            changed_projection_rows=[[1,1,2048],[1,1,0]],small_m_unchanged=True,m128_unchanged=True,
            parallel_stream_events_exact=True,all_new_gate_failures_raise=True,rejected_words=[])
(H/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
