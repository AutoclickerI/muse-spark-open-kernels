"""Extract actual post-exec S18 attention and apply the measured R8/D edits."""
import difflib,hashlib,importlib.util,json,sys
from pathlib import Path
import pvsum_build
from port_s18 import apply
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'fixed';o.mkdir(exist_ok=True)
p=R/'problems/muse_decode_layer/cand_r3_best7_s18_final.py'
spec=importlib.util.spec_from_file_location('p5b_s18_fixed_base',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
q=m.ATTN_PORT_CUDA_SRC
q=apply(q,(H/'page_ring/R8_S18.cuda.patch').read_text())
v=pvsum_build.cuda(m.VQ8_CUDA)
keys=('VQ8_CPP','ATTN_PORT_CPP_SRC','PARTIAL_CUDA','PARTIAL_CPP')
bundle={k:getattr(m,k) for k in keys};bundle.update(source=p.read_text(),q1=q,vq=v)
(o/'base_bundle.json').write_text(json.dumps(bundle))
for k in ('VQ8_CUDA','ATTN_PORT_CUDA_SRC','PARTIAL_CUDA'):(o/(k+'.cu')).write_text(getattr(m,k))
meta=dict(file=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),q1_sha256=hashlib.sha256(q.encode()).hexdigest(),vq_sha256=hashlib.sha256(v.encode()).hexdigest(),
          PP=repr(m._PP_ROWS),VQ=repr(m._VQ8_ROWS),settings=repr(m.BEST7_SETTINGS))
(o/'base_meta.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta)
