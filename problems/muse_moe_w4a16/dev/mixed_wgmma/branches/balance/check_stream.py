"""Audit the unique-byte reader against the exact packed-bank XOR."""
import json,runpy
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent
n=runpy.run_path(str(H/'balance_payload.py'),init_globals={'lab_cold_l2':lambda:None})
n['balance_build']();w=n['dc_weights'](5353,'cuda');s=n['balance_prepare'](w);d=n['dc_fixtures'](w,33)[0]
ids=torch.full_like(d['topk_ids'],-1);ids[0,0]=5;d=dict(d,topk_ids=ids)
g=n['oracle_graph'](d,s,0,32,False);out=g.replay();torch.cuda.synchronize();N=torch.cuda.get_device_properties(0).multi_processor_count*8*256
actual=int(np.bitwise_xor.reduce(out[:N].cpu().numpy().view(np.uint32)))
expected=0
for role in ('w13p','w2p'):expected^=int(np.bitwise_xor.reduce(s['native'][role][5].view(torch.int32).cpu().numpy().view(np.uint32).reshape(-1)))
assert actual==expected,(actual,expected)
result=dict(checksum=actual,expected=expected,work=g.work,expert=5,valid_tokens=1,unique_bytes=28311552,exact=True)
(H/'stream_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
