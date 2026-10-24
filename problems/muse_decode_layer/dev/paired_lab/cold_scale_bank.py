"""Measure a cold payload setup separately from the already-screened host."""
from pathlib import Path
import hashlib
import json
import linecache
import os
import sys
import tempfile
import time

import torch

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_decode_layer/dev'),
              str(ROOT/'problems/muse_decode_layer')]
import local_eval
import verify_best

host=local_eval.load_module('bank_cold_host',ROOT/'problems/muse_decode_layer/cand_r3_v9.py')
ref=local_eval.load_module('bank_cold_reference',ROOT/'problems/muse_decode_layer/reference.py')
weights=ref.generate_weights(ref.WEIGHTS_SEED)
state=host.prepare(weights)
source=(ROOT/'prof/lab/decode_scale_bank.py').read_text()
filename='<decode bank cold>'
linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
ns=dict(host.__dict__);exec(compile(source,filename,'exec'),ns)
ns['DP_LOCAL_INSTALL']=lambda m:verify_best.install(m,False)

# Check the board recode's represented values across every finite scale/code,
# including negative, zero and subnormal scales, before using its loaded bank.
raw=torch.arange(256,device='cuda',dtype=torch.uint8)
raw=raw[(raw&127)!=127]
sf=raw.contiguous().view(torch.float8_e4m3fn)
u,neg,zero=host._encode_scales(sf)
lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
codes=torch.arange(16,device='cuda')[None,:].expand(raw.numel(),16)
folded=torch.where(neg[:,None],codes^8,codes)
folded=torch.where(zero[:,None],torch.zeros_like(folded),folded)
scale=torch.exp2((u.int()>>3).float()-21)*(1+(u.int()&7).float()/8)
assert torch.equal(lut[folded]*scale[:,None],lut[codes]*sf.float()[:,None])

with tempfile.TemporaryDirectory(prefix='decode_scale_cold_') as cache:
    os.environ['TORCH_EXTENSIONS_DIR']=str(Path(cache)/'extensions')
    os.environ['TRITON_CACHE_DIR']=str(Path(cache)/'triton')
    os.environ['CUDA_CACHE_PATH']=str(Path(cache)/'cuda')
    started=time.monotonic();lines=ns['lab'](started+80.,weights,state)
    elapsed=time.monotonic()-started
rows=ns['DP_ROWS']
assert len(rows)==24 and not any('INCOMPLETE' in line for line in lines),lines
assert elapsed<80.,elapsed
result=dict(payload_sha256=hashlib.sha256(source.encode()).hexdigest(),elapsed_s=elapsed,
            finite_scales=raw.numel(),fp4_codes=16,recode_represented_values_exact=True,
            rows=rows,lines=lines)
(HERE/'scale_bank_cold_local.json').write_text(json.dumps(result,indent=2)+'\n')
print('\n'.join(lines))
print('COLD_SCALE_PASS',elapsed,'seconds; all 254 x 16 scale/code values exact')
