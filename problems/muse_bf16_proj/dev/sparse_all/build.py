"""Self-contained all-role codec extensions onto exact S15; no active layer edits."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[3];base=R/'problems/muse_decode_layer/cand_r3_best7_s15.py'
s=base.read_text();assert hashlib.sha256(s.encode()).hexdigest().startswith('60254883')
body=(H/'codec.py').read_text()+'\nDC_NATIVE = '+repr((H/'native.cuh').read_text())+'\n'
for name in ('triton.py.inc','patches.py.inc','adapter.py.inc','library.py.inc','ol.py.inc','board.py.inc'):
 if (H/name).exists():body+='\n'+(H/name).read_text()
out=s.replace('# okc: problem=muse_decode_layer','# okc: problem=muse_bf16_proj',1)+'\n'+body;compile(out,'<all role codec>','exec');assert len(out.encode())<1000000,len(out)
(H/'candidate.py').write_text(out);(H/'extension.py').write_text(body)
print('candidate',len(out),hashlib.sha256(out.encode()).hexdigest()[:8])
