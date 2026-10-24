"""Build the measured CTA1g2112 M256 path on the frozen S1 source."""
from pathlib import Path
import ast, hashlib, json, sys
R=Path(__file__).resolve().parents[4]
H=Path(__file__).parent
base=R/'problems/muse_decode_layer/cand_r3_best7_scale.py'
payload=R/'prof/lab/moe_w2fix_m256.py'
s=base.read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7'
p=payload.read_text();tree=ast.parse(p)
names={'mw_repack_planes','fx_prepare','fx_from_packed','fx_plan','fx_work','fx_call'}
functions='\n\n'.join(ast.get_source_segment(p,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('FX_CUDA','FX_CPP','FX_CONFIGS')}
s=s.replace("'humming_masked', 'auto'),", "'w2fix_cta1g2112', 'auto'),")
assert s.count("'w2fix_cta1g2112', 'auto'),")==2
s=s.replace("if moe not in ('auto',*_BEST6_CHOICES):", "if moe not in ('auto',*_BEST6_CHOICES,'w2fix_cta1g2112'):")
s=s.replace("def _best6_call(data,state,name):\n", "def _best6_call(data,state,name):\n    if name=='w2fix_cta1g2112':return fx_call(data,state['s2_bank'],1,2112,1)\n",1)
s=s.replace("def _best7_gate_moe(name,tokens,length,state,weights):\n", "def _best7_gate_moe(name,tokens,length,state,weights):\n    if name=='w2fix_cta1g2112':return _s2_gate_moe(tokens,length,state,weights)\n",1)
s=s.replace('with ThreadPoolExecutor(max_workers=8) as native_pool:', 'with ThreadPoolExecutor(max_workers=9) as native_pool:',1)
s=s.replace('        recoded_future=native_pool.submit(_v9_recoded_module)', '        recoded_future=native_pool.submit(_v9_recoded_module)\n        s2_future=native_pool.submit(fx_module)',1)
s=s.replace("        state['moe_recoded_mod']=recoded_future.result()", "        state['moe_recoded_mod']=recoded_future.result()\n        state['s2_mod']=s2_future.result()",1)
s=s.replace('    _best7_prepare_settings(state,weights,device)', "    s2_started=time.monotonic()\n    state['s2_bank']=fx_prepare(weights)\n    torch.cuda.synchronize(device)\n    phases['s2_repack']=time.monotonic()-s2_started\n    _best7_prepare_settings(state,weights,device)",1)
# Explicit table choices are mandatory. Screen failure stops candidate admission;
# it must never silently change M256 to another engine.
s=s.replace("                    if moe.startswith('v9nt'):raise", "                    if moe.startswith('v9nt') or moe=='w2fix_cta1g2112':raise",1)
s+='\n\n# H200 97be11d4631b44d6: exact raw-bank CTA1g2112 producer and finalizer.\n'
s+='import functools\n'+ '\n'.join(k+'='+repr(v) for k,v in constants.items())+'\n'
s+=(H/'runtime.py.inc').read_text()+'\n'+functions+'\n'
out=base.with_name('cand_r3_best7_scale_w2fix.py');out.write_text(s)
sys.path.insert(0,str(R/'tools'));from okc_run import rejected_words
assert not rejected_words(s),rejected_words(s)
manifest=dict(source=str(out.relative_to(R)),sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(p.encode()).hexdigest(),fx_cuda_sha256=hashlib.sha256(constants['FX_CUDA'].encode()).hexdigest(),receipt='97be11d4631b44d6',variant='CTA1g2112',config=dict(which=1,grid=2112,mode=1),functions=sorted(names))
(H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
