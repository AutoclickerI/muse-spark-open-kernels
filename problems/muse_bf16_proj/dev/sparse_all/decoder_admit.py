"""Pin the prepared follow-up before any outcome-dependent judge send."""
from pathlib import Path
import json,hashlib,ast
H=Path(__file__).resolve().parent
def read(name):return json.loads((H/name).read_text())
def sha(name):return hashlib.sha256((H/name).read_bytes()).hexdigest()
checks={}
for label in ('vector','view'):
    manifest=read(label+'_manifest.json');candidate=sha(label+'_candidate.py');assert manifest['candidate_sha256']==candidate
    for mode in ('normal','forced'):
        proof=read('screen/'+label+'_'+mode+'/result.json');assert proof['sha256']==candidate
        for key,count in (('tests',9),('shapes',30),('static',25)):
            assert len(proof[key])==count and all(r['ok'] for r in proof[key]),(label,mode,key)
        checks[label+'_'+mode]=dict(candidate_sha256=candidate,tests=9,shapes=30,inputs=3,static=25)
    proof=read(label+'_path_validation.json');assert proof['passed'] and proof['sha256']==candidate
    checks[label+'_paths']=dict(rows=len(proof['rows']),sha256=sha(label+'_path_validation.json'))
    proof=read(label+'_reader_validation.json');assert proof['passed'] and len(proof['rows'])==14
    checks[label+'_readers']=dict(rows=14,sha256=sha(label+'_reader_validation.json'))
    proof=read(label+'_native_compile.json');assert len(proof)==10 and all(r['ok'] for r in proof)
    assert all(all(set(v)=={'0'} for v in r.get('stack_spills',[])) for r in proof)
    checks[label+'_native']=dict(units=9,triton=1,sha256=sha(label+'_native_compile.json'))
manifest=read('decoder_manifest.json');proof=read('decoder_validation.json')
assert proof['passed'] and proof['payload_sha256']==manifest['payload_sha256']==sha('p10_decoder_followup.py')
assert proof['lines'][-1].startswith('END complete=10/10') and sum(len(x)+1 for x in proof['lines'])<9000
checks['rehearsal']=dict(sha256=sha('decoder_validation.json'),rows=10,seeds=8)
out=dict(passed=True,payload_sha256=manifest['payload_sha256'],checks=checks,
    decision='Send after a valid pool/all-role receipt. Compare same-bank scalar/vector/hoisted-view, not a whole-layer promotion. If a receipt fails setup/control, repair that implementation and preserve the valid control before using its timings.')
if (H/'decoder_host.py').exists():
    tree=ast.parse((H/'decoder_host.py').read_text());embedded=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets))
    assert len(embedded)==1 and hashlib.sha256(embedded[0][2].encode()).hexdigest()==manifest['payload_sha256'],'stale wrapper'
    out['wrapper_sha256']=sha('decoder_host.py');out['wrapper_bytes']=(H/'decoder_host.py').stat().st_size;assert out['wrapper_bytes']<1000000
(H/'decoder_admission.json').write_text(json.dumps(out,indent=2)+'\n');print('DECODER_ADMITTED',out['payload_sha256'][:8])
