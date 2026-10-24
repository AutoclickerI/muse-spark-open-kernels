"""Freeze best7 and generate the single self-contained host-independent lab."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_paired.py'
best7=source.read_text()
receipt=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best6.1b80ffc037cc454c.log'
choices=[]
for line in receipt.read_text().splitlines():
    if not line.startswith('BEST_SHAPE '):continue
    b,q,w=map(int,re.search(r'b=(\d+) q=(\d+) window=(\d+)',line).groups())
    if b*q>32:continue
    attn=ast.literal_eval(line.split('attn=',1)[1].split(' proj=',1)[0])
    proj=ast.literal_eval(line.split(' proj=',1)[1].split(' prefix=',1)[0])
    choices.append((b,q,w,attn,proj))
assert len(choices)==8
runtime='PP_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'projection.py').write_text('"""PDL dense projection drop-in; constructs only before graph capture."""\n'+runtime)
body='"""Whole-layer PDL weight prologues and many-CTA dense GEMV, one lab."""\n'
body+='PP_BEST7_SOURCE='+repr(best7)+'\nPP_CHOICES='+repr(choices)+'\n'+runtime+'\n'+(HERE/'lab.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(body),rejected_words(body)
out=ROOT/'prof/lab/dense_pdl_prefetch.py';out.write_text(body)
manifest=dict(payload=str(out.relative_to(ROOT)),sha256=hashlib.sha256(body.encode()).hexdigest(),
    baseline=str(source.relative_to(ROOT)),baseline_sha256=hashlib.sha256(best7.encode()).hexdigest(),
    kernel_sha256=hashlib.sha256((HERE/'kernel.cu').read_bytes()).hexdigest(),receipt=receipt.name,
    choices=choices)
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out,manifest['sha256'])
