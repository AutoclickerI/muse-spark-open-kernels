from pathlib import Path
import hashlib,json,ast
H=Path(__file__).resolve().parent;R=H.parents[3];base=R/'problems/muse_decode_layer/cand_r3_best7_s15.py';src=base.read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('60254883')
s=src+'\n'+(H/'layer_patch.py.inc').read_text();compile(s,'<pool layer>','exec');assert len(s.encode())<1000000
(H/'cand_s15_pool_raw.py').write_text(s)
(H/'layer_manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(src.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),tokens=[1,8,16],stage_job='bd3f8fc778ca4832'),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
