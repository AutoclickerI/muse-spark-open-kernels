import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_text();assert hashlib.sha256(base.encode()).hexdigest().startswith('60254883')
body='SGO_HELPER='+repr((H/'o1_epilogue.cuh').read_text())+'\n'+(H/'o1.py.inc').read_text();s=base+'\n'+body;compile(s,'<o1fused>','exec');assert len(s.encode())<1000000
(H/'o1_candidate.py').write_text(s);(H/'o1_extension.py.inc').write_text(body);(H/'o1_manifest.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),base='60254883',arms=[0,4,2,1]),indent=2)+'\n');print('o1 candidate',len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
