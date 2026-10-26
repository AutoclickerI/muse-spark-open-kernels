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
# Keep validated preceding cells if a later cell fails; emit the error and
# incomplete marker explicitly. An incomplete batch cannot promote a cell.
body=body.replace('def lab(deadline,weights,state=None):','def _dc_all_lab(deadline,weights,state=None):')
body=body.replace('    lines=[f\'META','    global DC_PROGRESS\n    lines=DC_PROGRESS=[f\'META')
body='\n'.join(line for line in body.splitlines() if not line.strip().startswith(("for name,s in st['stats']", "for (t,name),s in sorted(st['nf_sparse']")))+'\n'
body+='''
DC_PROGRESS=[]
def lab(deadline,weights,state=None):
    import traceback
    try:return _dc_all_lab(deadline,weights,state)
    except Exception as error:
        return DC_PROGRESS+[f'LAB_ERROR {type(error).__name__}: {error}']+traceback.format_exc().splitlines()[-8:]+['INCOMPLETE all-role gate failed; unmeasured']
'''
s=notice+'\nimport torch,base64,zlib\nDC_SOURCE=zlib.decompress(base64.b64decode('+repr(encoded)+')).decode()\n'+(H/'route_gate.py.inc').read_text()+'\n'+body
compile(s,'<all roles retry>','exec');(H/'p10_all_roles_retry.py').write_text(s)
(H/'all_retry_manifest.json').write_text(json.dumps(dict(payload_sha256=hashlib.sha256(s.encode()).hexdigest(),bytes=len(s.encode()),comparisons=57,direct=17,seeds=8,original='936dd5d5b1614711',change='quiet compact host and independently validated TASK routing score gate'),indent=2)+'\n');print(len(s),hashlib.sha256(s.encode()).hexdigest()[:8])
