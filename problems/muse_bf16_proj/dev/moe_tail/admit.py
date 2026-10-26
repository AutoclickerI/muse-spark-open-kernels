"""Only exact screened source and complete paired-control evidence may dispatch."""
import hashlib,json,lzma,pathlib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];manifest=json.loads((H/'manifest.json').read_text());base=(R/'problems/muse_decode_layer/cand_r3_best7_s15.py').read_bytes();assert hashlib.sha256(base).hexdigest()==manifest['s15_sha256']
for arm,sha in manifest['variants'].items():
 code=(H/(arm+'_trial.py')).read_bytes();assert code.startswith(base+b'\n') and hashlib.sha256(code).hexdigest()==sha
screens=json.loads(lzma.decompress((H/'screens.json.xz').read_bytes()))
for mode in ('normal','forced'):
 x=screens['screen_'+mode+'.json'];assert x['manifest']==manifest and len(x['arms'])==9
 for arm,r in x['arms'].items():assert len(r['tests'])==13 and len(r['shapes'])==12 and all(v['ok'] for v in r['tests']+r['shapes']),(mode,arm)
for arm,records in screens['python_validation.json'].items():assert records['sha256']==manifest['variants'][arm] and all(v['ok'] for v in records['checks'])
x=json.loads((H/'native_validation.json').read_text());assert x['source']==manifest and x['checks']==360
for kind in ('native','pmz'):
 x=json.loads((H/'native_audit.json').read_text())[kind];assert x['source_sha256']==manifest[kind+'_sha256']
 assert all('STRONG.SM' in l for k in x['kernels'] for l in k['before_wait'])
x=json.loads((H/'batch_validation.json').read_text());assert x['cases']==4 and x['seeds']==32 and x['arms']==9 and x['archive_chars']<8500 and x['all_gates_pass']
code=(H/'lab_moe_tail.py').read_bytes();m=json.loads((H/'lab_manifest.json').read_text());assert hashlib.sha256(code).hexdigest()==m['payload_sha256']
out=dict(**m,**manifest,normal_and_forced_tests=234,normal_and_forced_captured_shapes=216,native_checks=360,lab=x,native_ordering='Only immutable route metadata loads before wait; SS/Y volatile CG loads after wait; sm90a no local-memory SASS',host='muse_decode_layer S15 prepare phase',no_automatic_merge='Use the exact h200_lab wrapper from the lane out directory to preserve the full output/time budget')
if (H/'queue.json').exists():out['queue']=json.loads((H/'queue.json').read_text())
(H/'admission.json').write_text(json.dumps(out,indent=2)+'\n');print('ADMIT',m['payload_sha256'][:8],'all9 arms x normal/forced;4x32x8 balanced lab')
