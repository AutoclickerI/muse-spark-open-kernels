"""Explicit screen-only stand-in for S15's unrelated native N128 full-attention branch."""
import ast,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def install(host):
    src=(R/'problems/muse_decode_layer/cand_r3_best7_s14_final.py').read_text()
    assert hashlib.sha256(src.encode()).hexdigest()=='f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac'
    for n in ast.parse(src).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if getattr(t,'id','') in ('VQ8_CUDA','VQ8_CPP'):setattr(host,t.id,ast.literal_eval(n.value))
    host._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return 'Screen only: frozen S14 N64 full-attention arithmetic at S15 ST4 ABI; S15 native N128 is unchanged remotely'
