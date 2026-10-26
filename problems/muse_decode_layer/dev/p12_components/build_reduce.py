"""Keep measured P12 adapters frozen; append one output-ABI reducer."""
from pathlib import Path
import hashlib,json
H=Path(__file__).resolve().parent;s=(H/'cand_s15_projection.py').read_text()+'\n'+(H/'projection_reduce.py.inc').read_text()
compile(s,'<p12 reduced adapter>','exec');assert len(s.encode())<1000000
(H/'cand_s15_projection_reduce.py').write_text(s)
(H/'reduce_manifest.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),control_sha256=hashlib.sha256((H/'cand_s15_projection.py').read_bytes()).hexdigest()),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
