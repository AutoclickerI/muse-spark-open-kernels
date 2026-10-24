"""Freeze best6's real receipt and the side-stream transport in a single payload."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
base=ROOT/'problems/muse_decode_layer/cand_r3_best6.py'
source=base.read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528'
log=(base.parent/'profiles/cand_r3_best6.1b80ffc037cc454c.log').read_text()
choices=[];glue=[]
for line in log.splitlines():
    if line.startswith('BEST_SHAPE ') and ' q=1 ' in line:
        import re
        b,q,w=map(int,re.search(r'b=(\d+) q=(\d+) window=(\d+)',line).groups())
        attn=ast.literal_eval(line.split('attn=',1)[1].split(' proj=',1)[0])
        proj=ast.literal_eval(line.split(' proj=',1)[1].split(' prefix=',1)[0])
        choices.append((b,q,w,attn,proj))
    if line.startswith('GLUE_GEOMETRY '):
        row=ast.literal_eval(line.split(' ',1)[1])
        if row['t'] in (1,16,32):glue.append((row['t'],row['phase'],row['selected']))
assert len(choices)==6 and len(glue)==7  # log_tail dropped the first two M1 geometry lines
cuda=(HERE/'prefetch.cu').read_text()
transport='P7_PREFETCH_CUDA = '+repr(cuda)+'\n\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'prefetch.py').write_text('"""Graph-captured, within-call P7 prefetch transport."""\n'+transport)
out=HERE.parent/'l2_overlap_lab.py'
body='"""Paired within-call weight prefetch on the exact best6 decode dispatch."""\n'
body+='P7_BEST6_SOURCE = '+repr(source)+'\nP7_RECEIPT_CHOICES = '+repr(choices)+'\nP7_RECEIPT_GLUE = '+repr(glue)+'\n'
body+=transport+'\n\n'+(HERE/'lab.py.inc').read_text()
out.write_text(body)
(HERE/'manifest.json').write_text(json.dumps(dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),
    best6_sha256=hashlib.sha256(source.encode()).hexdigest(),receipt='1b80ffc037cc454c',choices=choices,glue=glue,
    cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest()),indent=2)+'\n')
print(out,hashlib.sha256(body.encode()).hexdigest())
