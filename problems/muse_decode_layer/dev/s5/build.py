"""Replace frozen S4's M128 engine with exact H_H from accepted 770809cf."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
BASE=R/'problems/muse_decode_layer/cand_r3_best7_s4.py'
WRAPPER=R/'runs/muse_moe_w4a16/20261024T155440Z_check_770809cf159f41e2.py'

def literal(s,key):
    return ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign)
                                  and any(getattr(t,'id','')==key for t in n.targets)))

def functions(s):
    lines=s.splitlines(keepends=True)
    return {n.name:''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno])
            for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}

payload=next(src for name,sha,src in literal(WRAPPER.read_text(),'_LAB_PAYLOADS') if name=='moe_m128_phase_mix')
payload_sha=hashlib.sha256(payload.encode()).hexdigest()
assert payload_sha=='f3c2796944c60b77aa5d63fd61e766a386bcdcfaf526b8ed01ab19e5367e5736'
cuda=literal(payload,'PM_CUDA')
# Exact independent native sections used by H_H: idist, pmnt, metadata.
dist_end=cuda.index('#include <cuda_runtime.h>',cuda.index('void idist_finish('))
nt_start=cuda.rfind('#include <cuda_runtime.h>',0,cuda.index('namespace pmnt {'))
native=cuda[:dist_end]+cuda[nt_start:]
assert all(name not in native for name in ('namespace norig','namespace or_rs'))
cpp='\n'.join(line for line in literal(payload,'PM_CPP').splitlines()
              if line.startswith(('#include','int64_t idist_','int64_t pm_nt_')))+'\n'
names=[n for n in literal(payload,'PM_FUNCTIONS') if n.startswith(('idist_','pm_'))]
assert len(names)==10
s=BASE.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='9b2b2a48337aa13ba1daa45eec8a30b566cc3e59c7e1bb317d39695413d3404e'
# The bank construction and descriptor encoding are already exact measured code.
fp,fb=functions(payload),functions(s)
def dump_function(text):return ast.dump(ast.parse(text),include_attributes=False)
assert dump_function(fp['issue_prepare'].replace('_encode_scales(', '_s4_encode_scales('))==dump_function(fb['issue_prepare'])
assert dump_function(fp['issue_word_bank'])==dump_function(fb['issue_word_bank'])
lines=s.splitlines(keepends=True)
for node in reversed(ast.parse(s).body):
    if isinstance(node,ast.FunctionDef) and node.name in ('issue_module','issue_plan','issue_call') or (
        isinstance(node,ast.Assign) and any(getattr(t,'id','') in ('S4_D32_CUDA','S4_D32_CPP') for t in node.targets)
    ):
        first=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
        del lines[first-1:node.end_lineno]
s=''.join(lines)
assert s.count('d32g_m128')==7,s.count('d32g_m128')
s=s.replace('d32g_m128','hh_m128')
for old,new in (
    ("return issue_call(data,state['s4_d32_bank'],'idist',1,1056)","return pm_call(data,state['s5_hh_base'],state['s5_hh_word'],state['moe_bank'],'H','H')"),
    ("s4_large_future=native_pool.submit(issue_module)","s5_large_future=native_pool.submit(pm_module)"),
    ("state['s4_large_mod']=s4_large_future.result()","state['s5_large_mod']=s5_large_future.result()"),
    ("state['s4_d32_bank']=issue_word_bank(issue_prepare(dict(weights,post_norm_w=weights['post_expert_norm'])))",
     "state['s5_hh_base']=issue_prepare(dict(weights,post_norm_w=weights['post_expert_norm']))\n    state['s5_hh_word']=issue_word_bank(state['s5_hh_base'])"),
    ("phases['s4_banks']", "phases['s5_banks']"),
    ("raise ValueError('S4 D32g geometry')", "raise ValueError('S5 H_H geometry')"),
):
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1)
s+='\n\n# H200 770809cf159f41e2: exact H_H expert/phase mix at M128.\n'
s+='S5_HH_CUDA='+repr(native)+'\nS5_HH_CPP='+repr(cpp)+'\n'
s+='S5_HH_FUNCTIONS='+repr(names)+'\n'
s+=(H/'runtime.py.inc').read_text()+'\n'
s+=fp['pm_plan']+'\n'+fp['pm_call']+'\n'
out=BASE.with_name('cand_r3_best7_s5.py');out.write_text(s)
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
manifest=dict(source=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),
              base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),receipt='770809cf159f41e2',
              accepted_wrapper=str(WRAPPER.relative_to(R)),payload_sha256=payload_sha,
              native_sha256=hashlib.sha256(native.encode()).hexdigest(),native_sections=['idist','pmnt','pm_meta'],
              variant='H_H',config=dict(first='H',second='H',small_expert_max_tokens=16,large_expert_min_tokens=17,
                                        nt1_stages=2,wgmma_which=1,wgmma_grid=1056),
              exact_functions=['pm_plan','pm_call','issue_prepare','issue_word_bank','mw_repack_planes'])
(H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
