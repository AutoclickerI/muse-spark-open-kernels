from pathlib import Path
import sys,subprocess,json
sys.path.insert(0,'problems/muse_prefill_layer/dev')
import canonical_dequant as d
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
P=Path('problems/muse_prefill_layer/dev');records=[]
for kb in (1024,2048):
 c=triton.compile(ASTSource(d._canonical_expand_coalesced,signature=dict(WB='*u8',WW='*i32',OUT='*i32',COUNTS='*i32',ROWS='constexpr',GATED='constexpr',KB='constexpr'),constexprs=dict(ROWS=8192,GATED=True,KB=kb)),target=GPUTarget('cuda',90,32),options=dict(num_warps=8))
 f=P/f'coalesced_dequant_{kb}_sm90.cubin';f.write_bytes(c.asm['cubin'])
 info=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(f)],text=True)
 print(kb,info,flush=True);records.append(dict(kb=kb,resources=info))
(P/'coalesced_dequant_sm90.json').write_text(json.dumps(records,indent=2))
