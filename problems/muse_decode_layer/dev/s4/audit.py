"""Prove accepted-payload provenance and the precise scope of the S4 integration."""
from pathlib import Path
import ast
import copy
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
m=json.loads((H/'manifest.json').read_text());source=R/m['source']
s=source.read_text();base=(R/'problems/muse_decode_layer/cand_r3_best7_scale_recoded_m256.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()==m['sha256']
assert hashlib.sha256(base.encode()).hexdigest()==m['base_sha256']
def funcs(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def literal(s,key):
    return ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign)
                                  and any(getattr(t,'id','')==key for t in n.targets)))
wrapper=(R/m['accepted_wrapper']).read_text()
payloads={name:src for name,sha,src in literal(wrapper,'_LAB_PAYLOADS')}
for name,sha in m['accepted_payloads'].items():assert hashlib.sha256(payloads[name].encode()).hexdigest()==sha
small=payloads['moe_nt1_limit'];large=payloads['moe_m128_issue_limit']
a,b,fs,fl=funcs(base),funcs(s),funcs(small),funcs(large)
changed={'prepare','_best6_call','_best7_gate_moe','_best7_prepare_settings'}
for name in a:
    if name not in changed:assert dump(a[name])==dump(b[name]),name
for name in m['exact_functions']:
    assert dump((fs if name=='nt1_call' else fl)[name])==dump(b[name]),name
class EncoderRename(ast.NodeTransformer):
    def visit_Name(self,n):
        if n.id=='_encode_scales':n.id='_s4_encode_scales'
        return n
expected=EncoderRename().visit(copy.deepcopy(fl['issue_prepare']))
assert dump(expected)==dump(b['issue_prepare'])
expected=copy.deepcopy(fl['_encode_scales']);expected.name='_s4_encode_scales'
assert dump(expected)==dump(b['_s4_encode_scales'])
# S3's shared NT1 bank exactly matches the measured repacker (docstring/name only differ).
def normalized_repacker(n):
    n=copy.deepcopy(n);n.name='pack'
    if isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant):n.body.pop(0)
    return n
assert dump(normalized_repacker(b['_repack_recoded']))==dump(normalized_repacker(fs['_repack']))
expected_encoder=copy.deepcopy(fs['_encode_scales'])
# S3 already handles an all-zero expert: only its validation guard differs.
guard=ast.parse('if bool(nz.any()):\n    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30\nassert int(mant.max()) <= 7').body
for index,node in enumerate(expected_encoder.body):
    if isinstance(node,ast.Assert) and any(isinstance(n,ast.Name) and n.id=='e5' for n in ast.walk(node)):
        expected_encoder.body[index:index+1]=guard
        break
else:raise AssertionError('expected measured scale-range validation')
assert dump(normalized_repacker(b['_encode_scales']))==dump(normalized_repacker(expected_encoder))
assert dump(normalized_repacker(b['_bit_table']))==dump(normalized_repacker(fs['_bit_table']))
assert literal(s,'S4_NT1_CUDA')==literal(small,'NT1_SOURCES')['lop3']
cuda=literal(large,'ISSUE_CUDA');start=cuda.rfind('#include <cuda_runtime.h>',0,cuda.index('namespace idist {'))
end=cuda.index('\n',cuda.index('void idist_finish('))
assert literal(s,'S4_D32_CUDA')==cuda[start:end]+'\n'
for key in ('RC_CUDA','RC_CPP','RC_CONFIGS','V9_CUDA_SRC','_V9_RECODED_CUDA_SRC','BEST7_DENSE_CHAIN','SP_CUDA','DENSE_CHAIN_CUDA'):
    assignment=lambda text:next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)
                                and any(getattr(t,'id','')==key for t in n.targets))
    assert dump(assignment(s))==dump(assignment(base)),key
old,new=literal(base,'BEST7_SETTINGS'),literal(s,'BEST7_SETTINGS')
for key,row in old.items():
    if key=='other' or key[0]*key[1]==256:assert new[key]==row
    else:assert new[key]==(*row[:2],'d32g_m128' if key[0]*key[1]==128 else 'nt1_lop3',row[3])
assert "issue_call(data,state['s4_d32_bank'],'idist',1,1056)" in s
assert "nt1_call(data,state['s4_nt1_bank'],'lop3',2,264)" in s
assert "moe in ('recoded_c1g2112','nt1_lop3','d32g_m128'):raise" in s
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
result=dict(sha256=m['sha256'],accepted_payloads_exact=True,small_m_kernel_exact=True,
            d32_kernel_exact_subsection=True,exact_functions=m['exact_functions'],
            unchanged_base_functions=len(a)-len(changed),changed_base_functions=sorted(changed),
            changed_rows=10,m256_unchanged=True,nt1_repacker_exact=True,
            scale_arithmetic_exact_with_s3_all_zero_guard=True,
            all_new_gate_failures_raise=True,rejected_words=[])
(H/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
