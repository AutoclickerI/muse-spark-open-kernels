"""Same full role coverage on a compact quiet host; preserve the lost receipt."""
from pathlib import Path
import hashlib,json,base64,zlib
H=Path(__file__).resolve().parent;R=H.parents[3]
notice=(R/'problems/muse_bf16_proj/dev/public_nf/nf_selected.py').read_text().split('import base64,zlib')[0]
encoded=base64.b64encode(zlib.compress((H/'candidate.py').read_bytes(),9)).decode();body=(H/'lab.py.inc').read_text()
body=body.replace("st['control']['router_bias']=T.zeros(256,device='cuda')","st['control']['router_bias']=weights.get('router_bias',.055*T.randn(256,device='cuda',generator=T.Generator(device='cuda').manual_seed(81901)))")
a=body.index("                a=saved[0][1]");b=body.index('\n            else:',a)
body=body[:a]+'''                for j,(_,out) in enumerate(saved):
                    e=dc_route_gate(out,x,weights['router'],st['control']['router_bias'],m['ROUTER_MULT'],m['RMS_EPS'],ref[:,:4096])
                    errors[j]=max(errors[j],e)
'''+body[b:]
s=notice+'\nimport torch,base64,zlib\nDC_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\n'+(H/'route_gate.py.inc').read_text()+'\n'+body
compile(s,'<all roles retry>','exec');(H/'p10_all_roles_retry.py').write_text(s)
(H/'all_retry_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),comparisons=57,direct=17,seeds=8,original='936dd5d5b1614711',change='quiet compact host and independently validated TASK routing score gate'),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
