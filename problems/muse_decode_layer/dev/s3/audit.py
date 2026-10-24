"""Pin S3 to S2 and the exact H200-measured recoded producer."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent
manifest = json.loads((H / 'manifest.json').read_text())
path = R / manifest['source']
s = path.read_text()
base = (R / 'problems/muse_decode_layer/cand_r3_best7_scale_w2fix.py').read_text()
payload = (R / manifest['payload']).read_text()
assert hashlib.sha256(s.encode()).hexdigest() == manifest['sha256']
assert hashlib.sha256(base.encode()).hexdigest() == manifest['base_sha256']
assert hashlib.sha256(payload.encode()).hexdigest() == manifest['payload_sha256']

def funcs(text):
    return {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

def dump(n):
    return ast.dump(n, include_attributes=False)

def literal(text, name):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == name for t in n.targets))
    return ast.literal_eval(node.value)

a, b, p = funcs(base), funcs(s), funcs(payload)
changed = {'prepare', '_best6_call', '_best7_gate_moe', '_best7_prepare_settings'}
removed = {'fx_module', '_s2_gate_moe', 'fx_prepare', 'fx_from_packed', 'fx_plan', 'fx_work', 'fx_call'}
for name in a:
    if name in removed:
        assert name not in b, name
    elif name not in changed:
        assert dump(a[name]) == dump(b[name]), name
for name in manifest['functions']:
    assert dump(p[name]) == dump(b[name]), name
for name in ('RC_CUDA', 'RC_CPP', 'RC_CONFIGS'):
    assert literal(s, name) == literal(payload, name), name
old, new = literal(base, 'BEST7_SETTINGS'), literal(s, 'BEST7_SETTINGS')
for key, value in old.items():
    if key in ((32, 8, 2048), (32, 8, 0)):
        assert new[key] == (*value[:2], 'recoded_c1g2112', value[3])
    else:
        assert new[key] == value, key
for name in ('BEST7_DENSE_CHAIN', 'V9_CUDA_SRC', '_pf_GROUPED_SRC', 'SP_CUDA', 'DENSE_CHAIN_CUDA'):
    assert literal(s, name) == literal(base, name), name
assert "state['s3_bank']=rc_prepare(weights)[1]" in s
assert "return rc_call(data,state['s3_bank'],1,2112)" in s
assert 's2_bank' not in s and 'FX_CUDA' not in s and 'w2fix_cta1g2112' not in s
sys.path.insert(0, str(R / 'tools'))
from okc_run import rejected_words
assert not rejected_words(s), rejected_words(s)

# The lab wrapper actually sent to H200 contains these same measured definitions.
receipt_path = R / 'runs/muse_moe_w4a16/20261024T150201Z_check_9e8a6a15f95748db.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['id'] == manifest['receipt'] and receipt['status'] == 'done'
assert receipt['result']['status'] == 'ok'
wrapper = receipt_path.with_suffix('.py').read_text()
accepted = next(row for row in literal(wrapper, '_LAB_PAYLOADS') if row[0] == 'moe_recoded_m256')
assert manifest['payload_sha256'].startswith(accepted[1]) and len(accepted[1]) == 16
assert accepted[2] == payload
w = funcs(accepted[2])
for name in manifest['functions']:
    assert dump(p[name]) == dump(w[name]), ('accepted wrapper', name)
for name in ('RC_CUDA', 'RC_CPP', 'RC_CONFIGS'):
    assert literal(accepted[2], name) == literal(payload, name), ('accepted wrapper', name)

result = dict(sha256=manifest['sha256'], unchanged_base_functions=len(a)-len(changed)-len(removed),
              changed_base_functions=sorted(changed), removed_raw_engine=sorted(removed),
              exact_measured_functions=manifest['functions'], native_source_exact=True,
              accepted_h200_wrapper_exact=True, changed_table_rows=[[32, 8, 2048], [32, 8, 0]],
              humming_transform_unchanged=True, s1_recoded_nt1_unchanged=True,
              retains_only_recoded_m256_bank=True, rejected_words=[])
(H / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
