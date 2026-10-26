"""Admit only the final exact source, full TASK screens and packed paired batch."""
import ast,hashlib,json,lzma,pathlib
H=pathlib.Path(__file__).resolve().parent;R=H.parents[3];m=json.loads((H/'manifest.json').read_text());screens=json.loads(lzma.decompress((H/'screens.json.xz').read_bytes()))
for arm,sha in m['variants'].items():assert hashlib.sha256((H/(arm+'_trial.py')).read_bytes()).hexdigest()==sha
for mode in ('normal','forced'):
 x=screens['screen_'+mode+'.json'];assert x['manifest']==m
 for arm,r in x['arms'].items():assert len(r['tests'])==13 and len(r['shapes'])==12 and all(v['ok'] for v in r['tests']+r['shapes']),(mode,arm)
for arm,r in screens['python_validation.json'].items():assert r['sha256']==m['variants'][arm] and all(v['ok'] for v in r['checks'])
v=json.loads((H/'native_validation.json').read_text());assert v['manifest']==m and v['mode_replays']==512 and v['denominator']==8192
assert json.loads((H/'native_audit.json').read_text())['source_sha256']==m['native_sha256']
b=json.loads((H/'batch_validation.json').read_text());assert b['TASK_checks']==640 and b['all_TASK_checks_pass'] and b['archive_chars']<8500
l=json.loads((H/'lab_manifest.json').read_text());assert hashlib.sha256((H/'lab_residual_tail.py').read_bytes()).hexdigest()==l['payload_sha256']
claim=json.loads((H/'claim_audit.json').read_text());assert claim['source_sha256']==m['s15_sha256'] and len(claim['confirmed'])==2 and not claim['dropped']
x=dict(**m,**l,normal_forced_test_cases=130,normal_forced_captured_shapes=120,native_mode_replays=512,batch=b,correctness='Copied public TASK checker every arm and fresh seed; no candidate-vs-baseline error or routing-membership veto',promotion='Every point ratio<1 goes to P4; held-out six-load score LCB>1 and TASK only')
(H/'admission.json').write_text(json.dumps(x,indent=2)+'\n');print('ADMIT',l['payload_sha256'][:8],b['archive_chars'],'chars;640 TASK checks;normal+forced pass')
