"""Freeze S19's actual post-exec attention namespace; no prepare or input data."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases';o.mkdir(exist_ok=True)
p=R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py';sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha.startswith('8b415553'),sha
spec=importlib.util.spec_from_file_location('p5_phases_s19',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
assert m._PP_ROWS==((1,1,False),) and m._PP_FAST is False
b={k:getattr(m,k) for k in ('VQ8_CPP','VQ8_CUDA','ATTN_PORT_CPP_SRC','ATTN_PORT_CUDA_SRC','PARTIAL_CUDA','PARTIAL_CPP')};b['source']=p.read_text()
(o/'base_bundle.json').write_text(json.dumps(b))
meta=dict(source=str(p.relative_to(R)),shipped=sha,expanded='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a',dispatch=dict(d1='native partial port; _PP_FAST=False',v1='FA3',d16d32='R8 attn_main',v16v32='D VQ8'))
(o/'base_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
