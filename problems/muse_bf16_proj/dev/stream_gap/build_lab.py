from pathlib import Path
import json,hashlib,zlib,base64
H=Path(__file__).resolve().parent;R=H.parents[3]
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
s=notice+'\nimport torch,base64,zlib\nSG_SOURCE=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress((H/'candidate.py').read_bytes(),9)).decode())+')).decode()\nSG_PROBE='+repr((H/'probe.cu').read_text())+'\n'+(H/'lab.py.inc').read_text()
compile(s,'<stream gap lab>','exec');(H/'payload.py').write_text(s);(H/'lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),source_sha256=hashlib.sha256((H/'candidate.py').read_bytes()).hexdigest()),indent=2)+'\n');print('payload',len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
