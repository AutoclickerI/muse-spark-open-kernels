"""Exact candidate, native unit, full wrapper and all paired TASK checks."""
from pathlib import Path
import json,hashlib,ast,re
H=Path(__file__).resolve().parent
def read(name):return json.loads((H/name).read_text())
def sha(name):return hashlib.sha256((H/name).read_bytes()).hexdigest()
manifest=read('layer_manifest.json');assert sha('cand_s15_pool_raw.py')==manifest['candidate_sha256']
lab=read('layer_lab_manifest.json');assert sha('pool_layer_payload.py')==lab['payload_sha256']
checks={}
for mode in ('normal','forced'):
    d=read('screen_layer/'+mode+'/result.json');assert d['sha256']==manifest['candidate_sha256']
    for key,n in (('tests',13),('shapes',12),('static',25)):assert len(d[key])==n and all(x['ok'] for x in d[key]),(mode,key)
    wrapper=read('screen_layer_lab/'+mode+'/result.json');assert wrapper['sha256']==sha('pool_layer_host.py')
    assert len(wrapper['tests'])==13 and all(x['ok'] for x in wrapper['tests']+wrapper['static'])
    p=read('screen_layer_lab/'+mode+'/lab.json');assert p['complete'] and p['lines'][-1]=='END complete=12/12 checks=768'
    assert len([x for x in p['lines'] if x.startswith('L ')])==12 and any(x.startswith('META checks=768/768') for x in p['lines'])
    checks[mode]=dict(candidate_tests=13,shapes=12,static=25,wrapper_tests=13,paired_shapes=12,paired_seeds=16,task_checks=768,
        wrapper_proof_sha256=sha('screen_layer_lab/'+mode+'/result.json'),paired_proof_sha256=sha('screen_layer_lab/'+mode+'/lab.json'))
native=read('layer_native.json');assert native['passed']
spills=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',native['log']);assert spills and all(set(x)=={'0'} for x in spills)
tree=ast.parse((H/'pool_layer_host.py').read_text());embedded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets))
assert len(embedded)==1 and hashlib.sha256(embedded[0][2].encode()).hexdigest()==lab['payload_sha256']
out=dict(passed=True,candidate_sha256=manifest['candidate_sha256'],payload_sha256=lab['payload_sha256'],wrapper_sha256=sha('pool_layer_host.py'),wrapper_bytes=(H/'pool_layer_host.py').stat().st_size,checks=checks,native=hashlib.sha256((H/'layer_native.json').read_bytes()).hexdigest(),gate='TASK and heldout six-load geomean lower95%>1; B816 vs C1816 decides optional M1')
assert out['wrapper_bytes']<1000000
(H/'layer_admission.json').write_text(json.dumps(out,indent=2)+'\n');print('LAYER_ADMITTED',out['candidate_sha256'][:8],out['wrapper_sha256'][:8])
