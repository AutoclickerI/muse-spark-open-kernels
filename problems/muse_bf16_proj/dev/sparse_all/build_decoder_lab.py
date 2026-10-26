"""One follow-up: all native-role decoder changes plus reader attribution."""
from pathlib import Path
import base64,zlib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[3]
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
encoded=base64.b64encode(zlib.compress((H/'candidate.py').read_bytes(),9)).decode()
s=notice+'\nimport torch,base64,zlib\nDC_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\n'
s+='FAST_SOURCE='+repr((H/'fast_decoder.py').read_text())+'\nSCAN_SOURCE='+repr((H/'decoder_scan.cu').read_text())+'\n'+(H/'decoder_lab.py.inc').read_text()
compile(s,'<decoder followup>','exec');assert len(s.encode())<1000000
(H/'p10_decoder_followup.py').write_text(s)
(H/'decoder_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),cells=10,arms=['S15','A2','scalar','vector','view'],seeds=8,scans=['raw','A2','scalar','vector','view','opaque']),indent=2)+'\n')
print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
