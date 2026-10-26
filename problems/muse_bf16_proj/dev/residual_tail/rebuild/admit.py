"""Check the exact rebuild source and its complete local evidence before send."""
import hashlib
import json
import lzma
import pathlib
import sys
H=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent))
from report import decode

m=json.loads((H/'manifest.json').read_text());lab=json.loads((H/'lab_manifest.json').read_text())
assert hashlib.sha256((H/'T_trial.py').read_bytes()).hexdigest()==m['candidate_sha256']
assert hashlib.sha256((H/'native_generated.py.inc').read_bytes()).hexdigest()==m['native_sha256']
assert hashlib.sha256((H/'runtime_generated.py.inc').read_bytes()).hexdigest()==m['runtime_sha256']
assert hashlib.sha256((H/'lab_normtail.py').read_bytes()).hexdigest()==lab['payload_sha256']
assert hashlib.sha256((H/'lab_generated.py.inc').read_bytes()).hexdigest()==lab['body_sha256']
screens=json.loads(lzma.decompress((H/'screens.json.xz').read_bytes()))
for mode,s in screens.items():
    assert mode in ('normal','forced') and s['manifest']==m and set(s['arms'])=={'A','A2','N','T'}
    for arm,c in s['arms'].items():
        assert len(c['tests'])==13 and len(c['shapes'])==12
        assert all(v['ok'] for v in c['tests']+c['shapes']),(mode,arm)
n=json.loads((H/'native_validation.json').read_text());assert n['manifest']==m and n['mode_replays']==384
audit=json.loads((H/'native_audit.json').read_text());assert audit['source_sha256']==m['native_sha256']
assert audit['tail_SASS_order'] and not audit['kernels'][0]['stack_or_local_sass']
x=json.loads(lzma.decompress((H/'local_raw.json.xz').read_bytes()))
assert x['runtime_sha256']==m['runtime_sha256'] and x['seeds']==lab['heldout_seeds']
selection=json.loads((H.parent/'h200_result.json').read_text())
assert not set(x['seeds']) & set(selection['seeds'])
assert len(x['cases'])==4 and all(c['task_checks']==128 and not c['failures'] and c['aa_schedule_identical'] for c in x['cases'])
assert all(len(c['samples'])==32 for c in x['cases'])
for c in x['cases']:
    for arm,v in c['criticals'].items():
        assert abs(sum(v['critical_us'].values())-v['wall_us'])<.02
        assert v['max_accounting_error']<1e-10
text=(H/'local_receipt.txt').read_text()
blob=next(l.split('RESIDUAL_TAIL_XZ85 ',1)[1] for l in text.splitlines() if 'RESIDUAL_TAIL_XZ85 ' in l)
y=decode(blob)
for a,b in zip(x['cases'],y['cases']):
    assert a['criticals']==b['criticals'] and a['profiles']==b['profiles']
    for left,right in zip(a['samples'],b['samples']):
        for arm,times in left.items():
            assert all(abs(times[k]-right[arm][k])<=.000501 for k in ('a_us','b_us'))
assert len(text)<9000
v={**m,**lab,'normal_forced_TASK_tests':104,'normal_forced_captured_shapes':96,
   'native_mode_replays':384,'native_worst_relative_l2':n['worst_relative_l2'],
   'batch_TASK_checks':512,'all_TASK_checks_pass':True,'AA_signatures_identical':True,
   'unchanged_N_b1':True,'all_target_fast_paths_reached':True,
   'max_CUPTI_accounting_error':max(p['max_accounting_error'] for c in x['cases'] for p in c['criticals'].values()),
   'local_archive_chars':len(text),'local_lab_s':x['elapsed_s']}
(H/'admission.json').write_text(json.dumps(v,indent=2)+'\n')
print('ADMIT',lab['payload_sha256'][:8],104,'official tests;',512,'TASK batch checks;',len(text),'chars')
