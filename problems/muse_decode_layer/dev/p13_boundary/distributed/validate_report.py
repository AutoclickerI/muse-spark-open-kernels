"""Check native-evidence attribution with synthetic transport, never a timing claim."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H),str(H.parent),str(R/'tools')]
import report as producer
from codec import pack_samples
from lab_packet import encode_packet

root=H/'out/report_validation';root.mkdir(exist_ok=True)
kind='boundary';manifest=json.loads((H/(kind+'_lab_manifest.json')).read_text())
(root/(kind+'_lab_manifest.json')).write_text(json.dumps(manifest))
packet=json.loads((H/'out/boundary_local_packet.json').read_text())
producer.H=root
def run(p,gpu):
    raw=dict(id='SYNTHETIC-NOT-A-NATIVE-RECEIPT',result=dict(gpu=gpu,passed=False,log_tail=encode_packet('p13_boundary',manifest['payload_sha256'][:16],pack_samples(copy.deepcopy(p)))))
    path=root/'receipt.json';path.write_text(json.dumps(raw));return producer.report(path,kind)
local=run(packet,'sm120 stand-ins')
assert not local['lab_winners'] and all(not v['score']['measured'] for v in local['candidates'].values())
bad=copy.deepcopy(packet);bad['failures'].append(dict(arm='B',label='injected',seed=bad['seed_ids'][0],nearby=False,reason='synthetic optional arm failure'))
attributed=run(bad,'H200')
assert attributed['candidates']['A']['score']['measured']
assert not attributed['candidates']['B']['score']['measured']
assert all(not v['official_promotion'] for v in attributed['candidates'].values())
positive=copy.deepcopy(packet)
for row in positive['paired']:
    row['round_ns']=[[[200000 if name=='INC' else 100000]*10 for name in positive['arms']] for _ in positive['seed_ids']]
(root/'out').mkdir(exist_ok=True)
for name in ('A','B'):
    (root/'out'/('boundary_'+name+'.inc')).write_text((H/'out'/('boundary_'+name+'.inc')).read_text())
qualified=run(positive,'H200')
assert qualified['lab_winners']==['A','B']
handoff=json.loads((root/'boundary_HANDOFF.json').read_text())
assert set(handoff['exact_patches'])=={'A','B'} and not handoff['official_promotion']
assert all(not v['official_promotion'] for v in qualified['candidates'].values())
wrong=copy.deepcopy(packet);wrong['seed_ids'][0]+=1
try:run(wrong,'H200')
except AssertionError:pass
else:raise AssertionError('wrong seed identity admitted')
print('PASS synthetic attribution only: stand-ins cannot qualify; optional failure isolated; exact winner patches retained; official TASK still required; identity mismatch rejected')
