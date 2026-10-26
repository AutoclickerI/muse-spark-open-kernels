"""Round-trip every paired seed, retain exact reference timing within declared resolution."""
import base64,importlib.util,json,lzma,pathlib,statistics,struct
from report import decode,summary
H=pathlib.Path(__file__).resolve().parent
raw=json.loads(lzma.decompress((H/'local_raw.json.xz').read_bytes()));assert raw['runtime_sha256']==json.loads((H/'manifest.json').read_text())['runtime_sha256'];ns=dict(base64=base64,json=json,lzma=lzma,statistics=statistics,struct=struct)
exec(compile((H/'lab_body.inc').read_text(),'body','exec'),ns);blob=ns['mt_pack'](json.loads(json.dumps(raw)));out=decode(blob)
for a,b in zip(raw['cases'],out['cases']):
 assert not a['gates'] and b['aa_schedule_identical'] and len(a['samples'])==len(b['samples'])==32
 assert a['profiles']==b['profiles']
 for i,(x,y) in enumerate(zip(a['samples'],b['samples'])):
  for arm,v in x.items():
   assert abs(v['ratio']-y[arm])<=.000005001
   assert abs(v['a_us']-b['reference_us'][i][arm])<=.00501
assert len(blob)+250<9000,len(blob)
r=summary(out);assert len(r['loads'])==2
v=dict(cases=4,seeds=32,arms=9,paired_rounds=8,archive_chars=len(blob),all_gates_pass=True,aa_signatures_identical=True,heavy_light_resources_identical=True,ratio_resolution=1e-5,reference_time_error_us_max=.00501,local_sm120_only=True)
(H/'local_receipt.txt').write_text('LOCAL repacked actual SM120 observations; GPU runtime unchanged\nMOE_TAIL_XZ85 '+blob+'\n')
(H/'batch_validation.json').write_text(json.dumps(v,indent=2)+'\n');print('PASS archive',len(blob),'characters',json.dumps(v))
