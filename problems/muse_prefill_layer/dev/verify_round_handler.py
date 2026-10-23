"""Test external-action decisions in a private fixture; never calls okc or alters submission.py."""
import datetime,hashlib,json,tempfile,types
from pathlib import Path
import finish_round as f
real_p=f.P
with tempfile.TemporaryDirectory(dir=real_p/'dev',prefix='round_test_') as td:
 root=Path(td);p=root/'owned';dev=p/'dev';dev.mkdir(parents=True)
 run=root/'runs/muse_prefill_layer';run.mkdir(parents=True);(root/'runs/pending').mkdir();(root/'runs/waiting').mkdir()
 ledger=root/'runs/ledger.jsonl';ledger.write_text('')
 original=(real_p/'submission.py').read_bytes();(p/'submission.py').write_bytes(original)
 f.ROOT=root;f.combo.ROOT=root;f.P=p;f.DEV=dev;f.STATE=dev/'state.json'
 f.combo.receipts=lambda:{}
 calls=[]
 f.launch=lambda mode,path,note:calls.append((mode,path,note)) or 123
 f.subprocess.run=lambda *a,**kw:types.SimpleNamespace(returncode=1,stdout='')
 a=(real_p/'cand_v17.py').read_bytes();b=(real_p/'cand_v15.py').read_bytes()
 state={'scoring':{},'integration':None,'allowed_sha':[hashlib.sha256(x).hexdigest() for x in (a,b)]}
 def receipt(job,blob,speed,flags=None):
  (run/f'test_check_{job}.py').write_bytes(blob)
  (run/f'test_check_{job}.json').write_text(json.dumps({'id':job,'result':{'passed':True,'speedup':speed,'flags':flags}}))
 receipt('aaa',a,1.5);f.audit(state,False)
 assert (p/'submission.py').read_bytes()==a and len(calls)==1
 f.audit(state,False);assert len(calls)==1
 receipt('bbb',b,1.6,['review']);f.audit(state,False)
 assert (p/'submission.py').read_bytes()==a and len(calls)==1
 receipt('bbb',b,1.6);f.audit(state,False)
 assert (p/'submission.py').read_bytes()==b and len(calls)==2
 # A higher passing check is promoted even when ten scored slots are already used;
 # it cannot start an eleventh scored run.
 state['scoring']={}
 ledger.write_text('\n'.join(json.dumps({'problem':'muse_prefill_layer','mode':'submit','job':str(i),'sha256':str(i),'utc':datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT000000Z')}) for i in range(10)))
 f.audit(state,False);assert len(calls)==2
 (run/'test_submit_failed.py').write_bytes(b)
 (run/'test_submit_failed.json').write_text(json.dumps({'id':'failed','result':{'passed':False}}))
 f.audit(state,False)
 assert (p/'submission.py').read_bytes()==a and len(calls)==2
 assert (real_p/'submission.py').read_bytes()==original
 print('PASS: exact checked source promotion; duplicate suppression; flags exclusion; quota cap; failed scored rerun rolls back; actual submission unchanged.')
