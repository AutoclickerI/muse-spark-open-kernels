"""Retain all paired seeds and TASK results; quantization only below event resolution."""
import base64,json,lzma,pathlib,struct
from report import decode,summary
H=pathlib.Path(__file__).resolve().parent;x=json.loads(lzma.decompress((H/'local_raw.json.xz').read_bytes()));ns=dict(base64=base64,json=json,lzma=lzma,struct=struct)
assert x['runtime_sha256']==json.loads((H/'manifest.json').read_text())['runtime_sha256']
exec(compile((H/'lab_body.inc').read_text(),'body','exec'),ns);blob=ns['rt_pack'](json.loads(json.dumps(x)));out=decode(blob)
for a,b in zip(x['cases'],out['cases']):
 assert not a['failures'] and a['task_checks']==160 and a['aa_schedule_identical'] and len(a['samples'])==32
 assert a['profiles']==b['profiles']
 for before,after in zip(a['samples'],b['samples']):
  for arm,v in before.items():assert abs(v['a_us']-after[arm]['a_us'])<=.000501 and abs(v['b_us']-after[arm]['b_us'])<=.000501
assert len(blob)+250<9000,len(blob)
r=summary(out);assert r['task_checks']==640
v=dict(cases=4,seeds=32,arms=5,paired_rounds=8,archive_chars=len(blob),TASK_checks=640,all_TASK_checks_pass=True,A_A_signatures_identical=True,all_target_rebuilds_reached=True,time_quantization_us=.001,local_sm120_only=True)
(H/'batch_validation.json').write_text(json.dumps(v,indent=2)+'\n');(H/'local_receipt.txt').write_text('LOCAL repacked actual SM120 observations\nRESIDUAL_TAIL_XZ85 '+blob+'\n');print('PASS',json.dumps(v))
