import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py'
base=source.read_text();allocation=(HERE/'allocation.py.inc').read_text()
runtime='MC_CUDA='+repr((HERE/'kernel.cu').read_text())+'\n'+(HERE/'runtime.py.inc').read_text()
body='MC_SOURCE='+repr(base)+'\n'+allocation+'\n'+runtime+'\n'+(HERE/'lab.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(body),rejected_words(body)
out=ROOT/'prof/lab/generic_memory_compression.py';out.write_text(body)
manifest=dict(payload=str(out.relative_to(ROOT)),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
    baseline=str(source.relative_to(ROOT)),baseline_sha256=hashlib.sha256(base.encode()).hexdigest(),
    allocation_sha256=hashlib.sha256(allocation.encode()).hexdigest(),
    kernel_sha256=hashlib.sha256((HERE/'kernel.cu').read_bytes()).hexdigest())
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(out,manifest['payload_sha256'])
