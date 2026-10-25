"""Pin final source, task checks, native builds and batch rehearsal for queueing."""
from pathlib import Path
import json,hashlib
H=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(name):return json.loads((H/name).read_text())
manifest=read('manifest.json');candidate=sha(H/'candidate.py');payload=sha(H/'p10_all_roles_lab.py')
assert candidate==manifest['candidate_sha256'] and payload==manifest['payload_sha256']
proofs={}
for arm in ('normal','forced'):
    x=read(f'screen/{arm}/result.json');assert x['sha256']==candidate
    for key,n in (('tests',9),('shapes',30),('static',25)):
        assert len(x[key])==n and all(r['ok'] for r in x[key]),(arm,key)
    proofs[arm]=dict(sha256=candidate,tests=9,shapes=30,inputs=3,static=25,prepare_s=x['prepare_s'],calls=read(f'screen/{arm}/calls.json'))
for name,key in (('codec_validation.json','candidate_sha256'),('path_validation.json','sha256'),('direct_validation.json','sha256'),('hopper_glue_validation.json','candidate_sha256')):
    x=read(name);assert x['passed'] and x[key]==candidate,name
    proofs[name]=dict(sha256=sha(H/name),rows=len(x['rows']))
x=read('reader_validation.json');assert x['passed'] and len(x['rows'])==14
proofs['reader_validation.json']=dict(sha256=sha(H/'reader_validation.json'),rows=14)
x=read('lab_validation.json');assert x['passed'] and x['payload_sha256']==payload
assert x['lines'][-1].startswith('END complete=57/57 direct=17')
assert sum(len(s)+1 for s in x['lines'])<9000
proofs['lab_validation.json']=dict(sha256=sha(H/'lab_validation.json'),rows=57,direct=17,local_seconds=x['elapsed'])
x=read('native_compile.json');assert len(x)==10 and all(r['ok'] for r in x)
assert all(all(set(v)=={'0'} for v in r.get('stack_spills',[])) for r in x)
x=read('direct_compile.json');assert len(x['rows'])==14 and all(r['ok'] for r in x['rows'])
for name in ('native_compile.json','direct_compile.json','sizes.json','native_sizes.json','load_bytes.json'):
    proofs[name]=dict(sha256=sha(H/name))
out=dict(passed=True,manifest=manifest,checks=proofs,wrapper_sha256=sha(H/'host_lab.py'),wrapper_bytes=(H/'host_lab.py').stat().st_size)
assert out['wrapper_bytes']<1000000
(H/'admission.json').write_text(json.dumps(out,indent=2)+'\n');print('ADMITTED',candidate[:8],payload[:8],out['wrapper_sha256'][:8])
