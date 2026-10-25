"""Paired S12 / PERMW / weight-reuse-order / combined lab, both M."""
import ast
import hashlib
import json
from pathlib import Path
import sys

H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import build
import patch_s12_permw
import patch_s12_reuse_order
from build_prefill_lab import subset


def value(s,key):
    return next(ast.literal_eval(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')==key)


def main():
    src=R/'problems/muse_decode_layer/cand_r3_best7_s12_final.py';s=src.read_text()
    assert hashlib.sha256(s.encode()).hexdigest()=='240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'
    perm=patch_s12_permw.patch(s)
    texts=[s,perm,patch_s12_reuse_order.patch(s),patch_s12_reuse_order.patch(perm)]
    cu=[value(t,'NF_CUDA') for t in texts]
    cpp=value(s,'NF_CPP')
    functions=build.strip_imports(subset(perm,('nf_plan','nf_call','nf_ilv','nf_wide','nf_perm')))
    body=build.helpers()+'\n'+build.IMPORTS+'\n'+build.strip_imports(build.HH_CELL.read_text())+'\n'+functions+'\n'
    body+=f'OP_CUDA={cu!r}\nOP_CPP={cpp!r}\nNF_OPTS={{}}\n'
    body+=(H/'order_runtime.py.inc').read_text()
    compile(body,'<order lab>','exec');(H/'order_payload.py').write_text(body)
    for i,c in enumerate(cu):(H/f'cu_order{i}.cu').write_text(c)
    manifest=dict(base_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  cuda_sha256=[hashlib.sha256(c.encode()).hexdigest() for c in cu])
    (H/'order_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)


if __name__=='__main__':main()
