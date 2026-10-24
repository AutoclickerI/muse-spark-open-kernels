"""Pin S5's M128 composition to the payload actually accepted as 770809cf."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
m=json.loads((H/'manifest.json').read_text());s=(R/m['source']).read_text()
base=(R/'problems/muse_decode_layer/cand_r3_best7_s4.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()==m['sha256']
assert hashlib.sha256(base.encode()).hexdigest()==m['base_sha256']
def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(node):return ast.dump(node,include_attributes=False)
def assignment(text,key):
    return next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
                and any(getattr(t,'id','')==key for t in n.targets))
def literal(text,key):return ast.literal_eval(assignment(text,key).value)
wrapper=(R/m['accepted_wrapper']).read_text()
name,short,payload=next(row for row in literal(wrapper,'_LAB_PAYLOADS') if row[0]=='moe_m128_phase_mix')
assert m['payload_sha256'].startswith(short) and hashlib.sha256(payload.encode()).hexdigest()==m['payload_sha256']
a,b,p=funcs(base),funcs(s),funcs(payload)
changed={'prepare','_best6_call','_best7_gate_moe','_best7_prepare_settings','_s4_prepare_banks','_s4_gate_moe'}
removed={'issue_module','issue_plan','issue_call'}
for name in a:
    if name in removed:assert name not in b,name
    elif name not in changed:assert dump(a[name])==dump(b[name]),name
for name in m['exact_functions']:
    expected=copy.deepcopy(p[name])
    if name=='issue_prepare':
        class Rename(ast.NodeTransformer):
            def visit_Name(self,n):
                if n.id=='_encode_scales':n.id='_s4_encode_scales'
                return n
        expected=Rename().visit(expected)
    assert dump(expected)==dump(b[name]),name
cuda=literal(payload,'PM_CUDA')
dist_end=cuda.index('#include <cuda_runtime.h>',cuda.index('void idist_finish('))
nt_start=cuda.rfind('#include <cuda_runtime.h>',0,cuda.index('namespace pmnt {'))
assert literal(s,'S5_HH_CUDA')==cuda[:dist_end]+cuda[nt_start:]
assert hashlib.sha256(literal(s,'S5_HH_CUDA').encode()).hexdigest()==m['native_sha256']
assert 'namespace norig' not in literal(s,'S5_HH_CUDA') and 'namespace or_rs' not in literal(s,'S5_HH_CUDA')
for key in ('RC_CUDA','RC_CPP','RC_CONFIGS','S4_NT1_CUDA','NT1_CPP','BEST7_DENSE_CHAIN',
            'V9_CUDA_SRC','_V9_RECODED_CUDA_SRC','SP_CUDA','DENSE_CHAIN_CUDA'):
    assert dump(assignment(s,key))==dump(assignment(base,key)),key
old,new=literal(base,'BEST7_SETTINGS'),literal(s,'BEST7_SETTINGS')
for key,row in old.items():
    if key in ((16,8,2048),(16,8,0)):assert new[key]==(*row[:2],'hh_m128',row[3])
    else:assert new[key]==row,key
assert "pm_call(data,state['s5_hh_base'],state['s5_hh_word'],state['moe_bank'],'H','H')" in s
assert "state['s5_hh_word']=issue_word_bank(state['s5_hh_base'])" in s
assert "moe in ('recoded_c1g2112','nt1_lop3','hh_m128'):raise" in s
assert 's4_d32_bank' not in s and 'S4_D32_CUDA' not in s and 'd32g_m128' not in s
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
result=dict(sha256=m['sha256'],accepted_payload_exact=True,native_sections_exact=True,
            exact_functions=m['exact_functions'],unchanged_base_functions=len(a)-len(changed)-len(removed),
            changed_base_functions=sorted(changed),removed_d32_engine=sorted(removed),
            changed_rows=[[16,8,2048],[16,8,0]],small_m_unchanged=True,m256_unchanged=True,
            shared_native_bank_unchanged=True,all_new_gate_failures_raise=True,rejected_words=[])
handoff=R/'problems/muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/hh_handoff.json'
if handoff.exists():
    cell=json.loads(handoff.read_text())
    assert cell['ready'] and cell['receipt']==m['receipt'] and cell['source_payload_sha256']==m['payload_sha256']
    assert hashlib.sha256((R/cell['path']).read_bytes()).hexdigest()==cell['sha256']
    for name,sha in cell['embedded_sources'].items():
        body=(handoff.parent/(name+'.cu')).read_text()
        assert hashlib.sha256(body.encode()).hexdigest()==sha
        assert body in literal(s,'S5_HH_CUDA'),name
    result['landed_moe_cell_crosscheck']=dict(sha256=cell['sha256'],same_native_bodies=True)
(H/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
