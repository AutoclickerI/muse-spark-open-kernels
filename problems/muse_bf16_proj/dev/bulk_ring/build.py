"""Bind one real-weight proj-host bulk-ring/O6 payload to frozen best7_scale."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];GF=HERE.parent/'gemv_fixed'
base=(ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7'
manifest=json.loads((GF/'manifest.json').read_text());choices={}
for key,value in manifest['choices'].items():
    t,role=key.split(':');choices[int(t),role]=tuple(value) if isinstance(value,list) else value
runtime='BR_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()
(HERE/'projection.py').write_text('"""Persistent bulk-ring projection, original BF16 weights."""\n'+runtime)
# Retain the already-measured O/config6 cell byte for byte in this batch.
fixed=(GF/'projection.py').read_text()
assert hashlib.sha256(fixed.encode()).hexdigest()=='c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe'
helpers=(GF/'lab.py.inc').read_text()
helpers=helpers[helpers.index('class GFSpace:'):helpers.index('def gf_probe(')]
# Replace synthetic base setup in the new payload; common graph/check/plan
# helpers remain source copies of our own lab, not imported reference code.
a=helpers.index('def gf_base(');b=helpers.index('def gf_plan(')
helpers=helpers[:a]+helpers[b:]
consumer=(GF/'diagnostic.py.inc').read_text();consumer=consumer[consumer.index('def gd_consumer('):consumer.index('def gd_summary(')]
body='GF_BASE_SOURCE='+repr(base)+'\nGF_PROJ='+repr(choices)+'\nGF_SEED=725108\n'
body+=fixed+'\n'+runtime+'\nimport gc\nimport linecache\nimport statistics\nimport time\n'+helpers+'\n'+consumer+'\n'+(HERE/'lab.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(body),rejected_words(body)
out=ROOT/'prof/lab/dense_bulk_ring.py';out.write_text(body)
record=dict(payload=str(out.relative_to(ROOT)),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
    projection=str((HERE/'projection.py').relative_to(ROOT)),projection_sha256=hashlib.sha256((HERE/'projection.py').read_bytes()).hexdigest(),
    kernel_sha256=hashlib.sha256((HERE/'kernel.cu').read_bytes()).hexdigest(),baseline_sha256=hashlib.sha256(base.encode()).hexdigest(),
    fixed_cell_sha256=hashlib.sha256(fixed.encode()).hexdigest(),host='muse_bf16_proj',weights='real host BF16/FP32; no synthetic weights',
    not_before='2026-10-25T09:02:00+09:00',choices=manifest['choices'])
(HERE/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(out,record['payload_sha256'])
