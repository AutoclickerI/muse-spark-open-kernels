import ast,hashlib,base64,zlib,json
from pathlib import Path
from component_source import projection_source
H=Path(__file__).resolve().parent;R=H.parents[3]
base=projection_source()
# Both changed mechanisms share one prepare and one judge payload.
body='SGO_HELPER='+repr((H/'o1_epilogue.cuh').read_text())+'\n'+(H/'o1.py.inc').read_text()+'\n'+(H/'reuse.py.inc').read_text()
s=base+'\n'+body;compile(s,'<projection followup>','exec');(H/'followup_candidate.py').write_text(s)
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
payload=notice+'\nimport torch,base64,zlib\nFOLLOW_SOURCE=zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(s.encode(),9)).decode())+')).decode()\n'+(H/'follow_route.py.inc').read_text()+'\n'+(H/'followup_lab.py.inc').read_text()
compile(payload,'<projection followup lab>','exec');(H/'p10_projection_followup.py').write_text(payload)
(H/'followup_manifest.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),bytes=len(payload.encode()),mechanisms=['last-CTA O1 residual+two norms','one M tile at256 avoids repeated weight loads; split K restores grid']),indent=2)+'\n');print(len(s),len(payload),hashlib.sha256(payload.encode()).hexdigest()[:8])
