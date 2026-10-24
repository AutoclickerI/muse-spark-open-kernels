"""S6: frozen S5 + accepted M256 mixed_parallel + the M1 O/config6 cell."""
from pathlib import Path
import ast
import hashlib
import json
import sys

R=Path(__file__).resolve().parents[4];H=Path(__file__).parent
BASE=R/'problems/muse_decode_layer/cand_r3_best7_s5.py'
WRAPPER=R/'runs/muse_moe_w4a16/20261024T161618Z_check_8483c0c6783a4df1.py'
PROJECTION=R/'problems/muse_bf16_proj/dev/gemv_fixed/projection.py'

def literal(s,key):
    return ast.literal_eval(next(n.value for n in ast.parse(s).body if isinstance(n,ast.Assign)
                                  and any(getattr(t,'id','')==key for t in n.targets)))

def functions(s):
    lines=s.splitlines(keepends=True)
    return {n.name:''.join(lines[min([n.lineno]+[d.lineno for d in n.decorator_list])-1:n.end_lineno])
            for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}

payload=next(src for name,sha,src in literal(WRAPPER.read_text(),'_LAB_PAYLOADS') if name=='moe_w13_issue_m256')
payload_sha=hashlib.sha256(payload.encode()).hexdigest()
assert payload_sha=='cb20858c5de9bfaca16ee289529cc2f46554ed575e32a7ce16dcccd31b1cd179'
projection=PROJECTION.read_text()
assert hashlib.sha256(projection.encode()).hexdigest()=='c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe'
s=BASE.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='f7064fc0127e80a955c7dc9aa5c68eb6aefada179c2841c5c58bce8dabd20284'
fp,fb=functions(payload),functions(s)
for name in ('mw_repack_planes','rc_masks','rc_repack'):
    assert ast.dump(ast.parse(fp[name]))==ast.dump(ast.parse(fb[name])),name

# Replace S3's M256 engine and its bank; retain the shared exact scale encoder.
removed={'rc_module','rc_prepare','rc_plan','rc_call','_s3_gate_moe'}
lines=s.splitlines(keepends=True)
for node in reversed(ast.parse(s).body):
    if isinstance(node,ast.FunctionDef) and node.name in removed or (
        isinstance(node,ast.Assign) and any(getattr(t,'id','') in ('RC_CUDA','RC_CPP','RC_CONFIGS') for t in node.targets)
    ):
        first=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
        del lines[first-1:node.end_lineno]
s=''.join(lines)
assert s.count('recoded_c1g2112')==6,s.count('recoded_c1g2112')
s=s.replace('recoded_c1g2112','mixed_parallel_m256')
for old,new in (
    ("return rc_call(data,state['s3_bank'],1,2112)","return wi_call(data,state['s6_bank'],6,2112)"),
    ("return _s3_gate_moe(tokens,length,state,weights)","return _s6_gate_moe(tokens,length,state,weights)"),
    ("s3_future=native_pool.submit(rc_module)","s6_future=native_pool.submit(fx_module)"),
    ("state['s3_mod']=s3_future.result()","state['s6_mod']=s6_future.result()"),
    ("s3_started=time.monotonic()","s6_started=time.monotonic()"),
    ("state['s3_bank']=rc_prepare(weights)[1]","state['s6_bank']=wi_prepare(weights)"),
    ("phases['s3_repack']=time.monotonic()-s3_started","phases['s6_repack']=time.monotonic()-s6_started"),
    ('def _best7_stream_op(tokens,name,choice,state,device):\n',
     "def _best7_stream_op(tokens,name,choice,state,device):\n    if (tokens,name,choice)==(1,'o',('stream',6,1)):return _s6_o_projection_op(state,device)\n"),
):
    assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new,1)
table=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign)
           and any(getattr(t,'id','')=='BEST7_SETTINGS' for t in n.targets))
settings=ast.literal_eval(table.value)
for key in ((1,1,2048),(1,1,0)):
    row=settings[key];assert row[3]=='auto'
    settings[key]=(*row[:3],{'o':('stream',6,1)})
lines=s.splitlines(keepends=True);lines[table.lineno-1:table.end_lineno]=['BEST7_SETTINGS = '+repr(settings)+'\n'];s=''.join(lines)

s+='\n\n# H200 8483c0c6783a4df1: exact M256 mixed_parallel policy6/grid2112.\n'
for name in ('FX_CUDA','FX_CPP','FX_CONFIGS','WI_CPP'):s+=name+'='+repr(literal(payload,name))+'\n'
s+=(H/'runtime.py.inc').read_text()+'\n'
names=('fx_prepare','fx_from_packed','fx_plan','wi_prepare','wi_plan','wi_call')
for name in names:s+=fp[name]+'\n'
s+='\n# H200 95ab383de42a4fae: role-local M1 O/config6 projection cell.\n'+projection+'\n'
out=BASE.with_name('cand_r3_best7_s6.py');out.write_text(s)
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
m=dict(source=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),
       base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),receipt='8483c0c6783a4df1',
       accepted_wrapper=str(WRAPPER.relative_to(R)),payload_sha256=payload_sha,
       native_sha256=hashlib.sha256(literal(payload,'FX_CUDA').encode()).hexdigest(),
       variant='mixed_parallel',config=dict(policy=6,grid=2112,small_expert_max_tokens=16,recoded_w13=True,recoded_w2=True),
       exact_functions=list(names),projection=dict(path=str(PROJECTION.relative_to(R)),
       sha256=hashlib.sha256(projection.encode()).hexdigest(),receipt='95ab383de42a4fae',
       tokens=1,role='o',config=6,splits=1,rows=[[1,1,2048],[1,1,0]],gain='near noise; whole-layer H200 gain unmeasured'))
(H/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
