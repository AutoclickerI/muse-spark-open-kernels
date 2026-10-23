from pathlib import Path
import importlib.util,subprocess,json
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
P=Path('problems/muse_prefill_layer');s=importlib.util.spec_from_file_location('swiglu_sm90',P/'cand_expert_portfolio.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
records=[]
for br,warps in ((1,4),(4,4),(4,8)):
 c=triton.compile(ASTSource(m._portfolio_swiglu,signature=dict(H='*bf16',A='*bf16',STARTS='*i32',COUNTS='*i32',GATE_GS='*fp32',UP_GS='*fp32',D='constexpr',BR='constexpr'),constexprs=dict(D=4096,BR=br)),target=GPUTarget('cuda',90,32),options=dict(num_warps=warps))
 f=P/'dev'/f'portfolio_swiglu_{br}_{warps}_sm90.cubin';f.write_bytes(c.asm['cubin'])
 r=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
 print(br,warps,r,flush=True);records.append(dict(br=br,warps=warps,resources=r))
(P/'dev/swiglu_sm90.json').write_text(json.dumps(records,indent=2))
