"""Pin S2 to the exact S1 base and the measured H200 producer."""
from pathlib import Path
import ast,hashlib,json,sys
R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
manifest=json.loads((H/'manifest.json').read_text());path=R/manifest['source']
s=path.read_text();base=(R/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text()
payload=(R/'prof/lab/moe_w2fix_m256.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()==manifest['sha256']
assert hashlib.sha256(base.encode()).hexdigest()==manifest['base_sha256']
assert hashlib.sha256(payload.encode()).hexdigest()==manifest['payload_sha256']
def funcs(text):return {n.name:n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
def literal(text,name):
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
    return ast.literal_eval(node.value)
a,b,p=funcs(base),funcs(s),funcs(payload)
changed={'prepare','_best6_call','_best7_gate_moe','_best7_prepare_settings'}
for name in a:
    if name not in changed:assert dump(a[name])==dump(b[name]),name
for name in manifest['functions']:assert dump(p[name])==dump(b[name]),name
for name in ('FX_CUDA','FX_CPP','FX_CONFIGS'):assert literal(s,name)==literal(payload,name),name
old=literal(base,'BEST7_SETTINGS');new=literal(s,'BEST7_SETTINGS')
for key,value in old.items():
    if key in ((32,8,2048),(32,8,0)):assert new[key]==(*value[:2],'w2fix_cta1g2112',value[3])
    else:assert new[key]==value,key
for name in ('BEST7_DENSE_CHAIN','V9_CUDA_SRC','_pf_GROUPED_SRC','SP_CUDA','DENSE_CHAIN_CUDA'):
    assert literal(s,name)==literal(base,name),name
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
result=dict(sha256=manifest['sha256'],unchanged_base_functions=len(a)-len(changed),changed_base_functions=sorted(changed),exact_measured_functions=manifest['functions'],native_source_exact=True,changed_table_rows=[[32,8,2048],[32,8,0]],humming_transform_unchanged=True,s1_recoded_nt1_unchanged=True,rejected_words=[])
(H/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
