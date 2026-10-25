import hashlib
from pathlib import Path
from patch import cuda_source,patch

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
base=(REPO/'problems/muse_decode_layer/cand_r3_best7_s13_final.py').read_text()
cuda=cuda_source(base);(HERE/'glue_preload.cu').write_text(cuda)
rows=tuple((b,q,w,role) for b,q in ((1,1),(16,1),(32,1),(1,8))
    for w in (2048,0) for role in ('b1','route','b2'))
src=patch(base,rows);out=HERE/'preload_trial.py';out.write_text(src)
print(out,hashlib.sha256(src.encode()).hexdigest())
