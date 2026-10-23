"""Record ATen dispatch of the released selector's strided hidden projection."""
import json
from pathlib import Path
import torch
from torch.nn.functional import linear

rows=[]
for c in (1,4,16,32):
    with torch.inference_mode():
        hidden=torch.randn((c,8,8192),device='cuda',dtype=torch.bfloat16)[:,1:]
        w=torch.nn.Parameter(torch.randn((256,8192),device='cuda',dtype=torch.bfloat16))
        original=linear(hidden,w);packed=linear(hidden.reshape(-1,8192),w).view(c,7,256)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            linear(hidden,w)
        ops=[e.key for e in prof.key_averages() if any(s in e.key for s in ('linear','mm','clone','copy','reshape'))]
        rows.append(dict(c=c,strides=hidden.stride(),operators=ops,flattening_exact=torch.equal(original,packed),
                         max_difference=float((original.float()-packed.float()).abs().max())))
path=Path('runs/screen/serving_dflash/selector_layout.json');path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
