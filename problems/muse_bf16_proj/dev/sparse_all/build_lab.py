"""Freeze the full candidate in a single self-contained H200 payload."""
from pathlib import Path
import base64,zlib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[3]
license=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
source=(H/'candidate.py').read_text();encoded=base64.b64encode(zlib.compress(source.encode(),9)).decode()
s=license+'\nimport torch,base64,zlib\nDC_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\n'+(H/'lab.py.inc').read_text()
compile(s,'<P10 all sparse roles>','exec');assert len(s.encode())<1000000
(H/'p10_all_roles_lab.py').write_text(s)
manifest=dict(base='problems/muse_decode_layer/cand_r3_best7_s15.py',base_sha256=hashlib.sha256((R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_bytes=len(source.encode()),payload_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_bytes=len(s.encode()),comparisons=57,direct_arms=17,seeds=8,cold_replays_per_seed=8)
(H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
