from pathlib import Path
import base64,zlib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[3]
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
encoded=base64.b64encode(zlib.compress((H/'cand_s15_projection.py').read_bytes(),9)).decode()
s=notice+'\nimport torch,base64,zlib\nPROJ_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\nREDUCE_SOURCE='+repr((H/'projection_reduce.py.inc').read_text())+'\n'+(H/'reduce_lab.py.inc').read_text()
compile(s,'<p12 reduce lab>','exec');(H/'p12_projection_reduce_lab.py').write_text(s)
(H/'reduce_lab_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),cells=30,seeds=8),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
