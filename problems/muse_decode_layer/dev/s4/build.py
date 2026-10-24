"""Assemble S4 from S3 and exact accepted dc3fea0a payload definitions."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
BASE=R/'problems/muse_decode_layer/cand_r3_best7_scale_recoded_m256.py'
WRAPPER=R/'runs/muse_moe_w4a16/20261024T150834Z_check_dc3fea0a20a349bf.py'

def literals(s):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(s).body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('_LAB_PAYLOADS','NT1_SOURCES','NT1_CPP','ISSUE_CUDA','ISSUE_CPP')}

def functions(s):
    lines=s.splitlines(keepends=True)
    return {n.name:''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno])
            for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}

payloads={name:(sha,src) for name,sha,src in literals(WRAPPER.read_text())['_LAB_PAYLOADS']}
small=payloads['moe_nt1_limit'][1];large=payloads['moe_m128_issue_limit'][1]
assert hashlib.sha256(small.encode()).hexdigest()=='7014f33df86522d3f99803f3af5458f3c5875f9813f73f9998d48af99df81b93'
assert hashlib.sha256(large.encode()).hexdigest()=='9ec548dc3883ffcdad5d2442e793e1bbf8e94b92733a68a78de56c8d1b1a6c0b'
vsmall,vlarge=literals(small),literals(large)
fs,fl=functions(small),functions(large)
cuda=vlarge['ISSUE_CUDA']
start=cuda.rfind('#include <cuda_runtime.h>',0,cuda.index('namespace idist {'))
finish=cuda.index('void idist_finish(')
end=cuda.index('\n',finish)
dist=cuda[start:end]+'\n'
assert 'namespace ibase' not in dist and 'namespace nlop' not in dist
cpp='\n'.join(line for line in vlarge['ISSUE_CPP'].splitlines()
              if line.startswith('#include') or line.startswith('int64_t idist_'))+'\n'
s=BASE.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7'
# Exact row edits: small M1/8/16/32 -> LOP3; M128 -> distributed D32g.
table=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign)
           and any(getattr(t,'id','')=='BEST7_SETTINGS' for t in n.targets))
settings=ast.literal_eval(table.value)
for key,row in list(settings.items()):
    if key=='other' or key[0]*key[1]==256:continue
    settings[key]=(*row[:2],'d32g_m128' if key[0]*key[1]==128 else 'nt1_lop3',row[3])
lines=s.splitlines(keepends=True)
lines[table.lineno-1:table.end_lineno]=['BEST7_SETTINGS = '+repr(settings)+'\n']
s=''.join(lines)
for old,new in (
    ('def _best6_call(data,state,name):\n',"def _best6_call(data,state,name):\n    if name=='nt1_lop3':return nt1_call(data,state['s4_nt1_bank'],'lop3',2,264)\n    if name=='d32g_m128':return issue_call(data,state['s4_d32_bank'],'idist',1,1056)\n"),
    ('def _best7_gate_moe(name,tokens,length,state,weights):\n',"def _best7_gate_moe(name,tokens,length,state,weights):\n    if name in ('nt1_lop3','d32g_m128'):return _s4_gate_moe(name,tokens,length,state,weights)\n"),
    ("if moe not in ('auto',*_BEST6_CHOICES,'recoded_c1g2112'):","if moe not in ('auto',*_BEST6_CHOICES,'recoded_c1g2112','nt1_lop3','d32g_m128'):"),
    ("if moe.startswith('v9nt') or moe=='recoded_c1g2112':raise","if moe.startswith('v9nt') or moe in ('recoded_c1g2112','nt1_lop3','d32g_m128'):raise"),
    ('ThreadPoolExecutor(max_workers=9)','ThreadPoolExecutor(max_workers=11)'),
    ('        s3_future=native_pool.submit(rc_module)', '        s3_future=native_pool.submit(rc_module)\n        s4_small_future=native_pool.submit(nt1_module,\'lop3\')\n        s4_large_future=native_pool.submit(issue_module)'),
    ("        state['s3_mod']=s3_future.result()","        state['s3_mod']=s3_future.result()\n        state['s4_small_mod']=s4_small_future.result()\n        state['s4_large_mod']=s4_large_future.result()"),
    ('    _best7_prepare_settings(state,weights,device)',"    s4_started=time.monotonic()\n    _s4_prepare_banks(state,weights)\n    torch.cuda.synchronize(device)\n    phases['s4_banks']=time.monotonic()-s4_started\n    _best7_prepare_settings(state,weights,device)"),
):
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1)
assert ast.dump(ast.parse(fl['mw_repack_planes']))==ast.dump(ast.parse(functions(s)['mw_repack_planes']))
s+='\n\n# H200 dc3fea0a20a349bf: exact one-LOP3 small-M and distributed D32g M128.\n'
s+='S4_NT1_CUDA='+repr(vsmall['NT1_SOURCES']['lop3'])+'\n'
s+='NT1_CPP='+repr(vsmall['NT1_CPP'])+'\n'
s+='S4_D32_CUDA='+repr(dist)+'\nS4_D32_CPP='+repr(cpp)+'\n'
s+='NT1_MODULES={}\nISSUE_OPTS={}\n'
s+=(H/'runtime.py.inc').read_text()+'\n'
s+=fs['nt1_call']+'\n'
s+=fl['_encode_scales'].replace('def _encode_scales(', 'def _s4_encode_scales(')+'\n'
s+=fl['issue_prepare'].replace('_encode_scales(', '_s4_encode_scales(')+'\n'
for name in ('issue_word_bank','issue_plan','issue_call'):s+=fl[name]+'\n'
out=BASE.with_name('cand_r3_best7_s4.py');out.write_text(s)
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
manifest=dict(source=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),
              base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),receipt='dc3fea0a20a349bf',
              accepted_wrapper=str(WRAPPER.relative_to(R)),
              accepted_payloads={n:hashlib.sha256(src.encode()).hexdigest() for n,(sha,src) in payloads.items()},
              nt1_cuda_sha256=hashlib.sha256(vsmall['NT1_SOURCES']['lop3'].encode()).hexdigest(),
              d32_cuda_sha256=hashlib.sha256(dist.encode()).hexdigest(),
              nt1=dict(kind='lop3',stages=2,grid=264,clamp_to_residency=True,tokens=[1,8,16,32]),
              m128=dict(prefix='idist',which=1,grid=1056,scale_word_bits=16),
              exact_functions=['nt1_call','issue_word_bank','issue_plan','issue_call','mw_repack_planes'],
              renamed_functions={'issue_prepare':'_encode_scales -> _s4_encode_scales',
                                 '_s4_encode_scales':'_encode_scales definition renamed'})
(H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
