"""Admission for the combined independent structural follow-up."""
from pathlib import Path
import json,hashlib,ast
H=Path(__file__).resolve().parent
def read(name):return json.loads((H/name).read_text())
def sha(name):return hashlib.sha256((H/name).read_bytes()).hexdigest()
decoder=read('decoder_admission.json');assert decoder['passed']
block=read('block_manifest.json');assert block['candidate_sha256']==sha('block_candidate.py')
checks={}
for mode in ('normal','forced'):
    proof=read('screen/block_'+mode+'/result.json');assert proof['sha256']==block['candidate_sha256']
    for key,n in (('tests',9),('shapes',30),('static',25)):assert len(proof[key])==n and all(x['ok'] for x in proof[key])
    checks[mode]=dict(tests=9,shapes=30,inputs=3,static=25)
for name in ('block_path_validation.json','block_direct_validation.json'):
    proof=read(name);assert proof['passed'] and proof['sha256']==block['candidate_sha256'];checks[name]=dict(rows=len(proof['rows']),sha256=sha(name))
proof=read('block_validation.json');assert proof['passed'] and len(proof['rows'])==10 and proof['source_sha256']==sha('block_reader.py.inc')
proof=read('block_compile.json');assert proof['candidate_sha256']==block['candidate_sha256'] and len(proof['rows'])==12 and all(x['ok'] for x in proof['rows'])
manifest=read('combined_manifest.json');proof=read('combined_validation.json');assert proof['passed'] and proof['payload_sha256']==manifest['payload_sha256']==sha('p10_coding_structural.py')
assert proof['lines'][-1].startswith('POINT_END complete=20/20') and any(x.startswith('END complete=10/10') for x in proof['lines'])
assert sum(len(x)+1 for x in proof['lines'])<9000
host=(H/'combined_host.py').read_text();tree=ast.parse(host)
embedded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets))
assert len(embedded)==1 and hashlib.sha256(embedded[0][2].encode()).hexdigest()==manifest['payload_sha256'];assert len(host.encode())<1000000
out=dict(passed=True,payload_sha256=manifest['payload_sha256'],wrapper_sha256=sha('combined_host.py'),wrapper_bytes=len(host.encode()),
    decoder_admission_sha256=sha('decoder_admission.json'),block_candidate_sha256=block['candidate_sha256'],checks=checks,
    cells=30,seeds=8,local_seconds=proof['seconds'],rehearsal_sha256=sha('combined_validation.json'))
(H/'combined_admission.json').write_text(json.dumps(out,indent=2)+'\n');print('COMBINED_ADMITTED',out['payload_sha256'][:8],out['wrapper_sha256'][:8])
