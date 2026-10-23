"""Freeze the source/log correspondence for the morning hand-off. No remote operations."""
from pathlib import Path
import ast
import hashlib
import json
import re
p=Path(__file__).resolve().parent.parent

def strings(file):
 answer={}
 for n in ast.parse(file.read_text()).body:
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
   for t in n.targets:
    if isinstance(t,ast.Name):answer[t.id]=n.value.value
  elif isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id in answer:
   answer[n.target.id]+=ast.literal_eval(n.value)
 return answer

logs={'cand_rs.py':'local_rs.log','cand_rs128.py':'local_rs128.log',
      'cand_rs7.py':'local_rs7.log','cand_rs128_7.py':'local_rs128_7.log',
      'cand_ready9.py':'local_ready9.log','cand_ready.py':'local_ready.log',
      'cand_retire.py':'local_retire.log','cand_probe.py':'local_probe_tma_final.log','cand_rs_tma.py':'local_rs_tma_final.log'}
manifest={}
for name,logname in logs.items():
 data=(p/name).read_bytes();values=strings(p/name)
 log=(p/'dev'/logname).read_text()
 assert len(re.findall(r'^  PASS ',log,re.M))==5,(name,'standard tests')
 assert len(re.findall(r'^  ok ',log,re.M))==6,(name,'graph timings')
 assert 'FAIL' not in log and 'Traceback' not in log,name
 manifest[name]={'source_sha256':hashlib.sha256(data).hexdigest(),'local_log':logname,
                 'cuda_sha256':hashlib.sha256(values['CUDA_SRC'].encode()).hexdigest()}
 if 'GROUPED_SRC' in values:
  manifest[name]['grouped_cuda_sha256']=hashlib.sha256(values['GROUPED_SRC'].encode()).hexdigest()
 if 'V7_CUDA_SRC' in values:
  manifest[name]['v7_cuda_sha256']=hashlib.sha256(values['V7_CUDA_SRC'].encode()).hexdigest()

for source,logname in (('cand_rs.py','compile_round2.log'),('cand_rs128.py','compile_rs128.log'),
                       ('cand_probe.py','compile_probe_tma_final.log'),('cand_rs_tma.py','compile_rs_tma_final.log'),('cand_ready9.py','compile_probe_ready9.log')):
 log=(p/'dev'/logname).read_text()
 expected=manifest[source].get('grouped_cuda_sha256',manifest[source]['cuda_sha256'])
 assert f'SM90a OFFLINE {source} {expected}' in log,(source,'offline source mismatch')
 assert f'OFFLINE COMPILE PASS {source}' in log,(source,'offline compile')
v7=strings(p/'cand_v7.py');probe=strings(p/'cand_probe.py')
assert probe['V7_CUDA_SRC']==v7['CUDA_SRC'] and probe['V7_CPP_SRC']==v7['CPP_SRC']
for source,base in (('cand_rs7.py','cand_rs.py'),('cand_rs128_7.py','cand_rs128.py')):
 s,b=strings(p/source),strings(p/base)
 assert s['CUDA_SRC']==v7['CUDA_SRC'] and s['CPP_SRC']==v7['CPP_SRC']
 assert s['GROUPED_SRC']==b['GROUPED_SRC']
 manifest[source]['native_receipt']='f95bdf5e394e474e'
# The canonical repacker is exactly the same source in the two checked native revisions.
v6_ast=ast.parse((p/'cand_v6.py').read_text());v7_ast=ast.parse((p/'cand_v7.py').read_text())
for name in ('_repack','_encode_scales','_bit_table','prepare'):
 a=next(n for n in v6_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
 b=next(n for n in v7_ast.body if isinstance(n,ast.FunctionDef) and n.name==name)
 assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False),name
for logname in ('stress_rs.log','stress_rs128.log'):
 log=(p/'dev'/logname).read_text()
 assert len(re.findall(r'^tile ',log,re.M))==12
 assert len(re.findall(r'^routing replay ',log,re.M))==4
 assert len(re.findall(r'^large ',log,re.M))==6
 assert 'Traceback' not in log
for m in (1,16,32,8,128,256):
 log=(p/'dev/local_probe_tma_final.log').read_text();marker=f'MUSE_SWEEP m={m} call=16 '
 assert marker in log,marker
 block=log.split(marker)[1].split('MUSE_SWEEP m=')[1:15]
 assert len(block)==14
 for i,text in enumerate(block):
  assert text.startswith(f'{m} variant={i} '),(m,i,text[:60])
  assert 'peer_l2=' in text and 'finite=1' in text,(m,i,'peer validation missing')
tma=strings(p/'cand_rs_tma.py')
assert tma['CUDA_SRC']==v7['CUDA_SRC'] and tma['CPP_SRC']==v7['CPP_SRC']
log=(p/'dev/stress_rs_tma_final.log').read_text()
assert len(re.findall(r'^N ',log,re.M))==9
assert len(re.findall(r'^WS2 graph routing ',log,re.M))==4
assert len(re.findall(r'^large ',log,re.M))==6
assert 'HOPPER_ONLY N 128' in log and 'Traceback' not in log
(p/'dev/round2_manifest.json').write_text(json.dumps(manifest,indent=2))
print('PASS: nine complete local screens; 14 graph intervals and peer checks per load; exact compiled/checked CUDA cores; identical canonical repacker; large-M suites.')
