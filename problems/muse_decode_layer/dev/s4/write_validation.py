"""Source-pinned normal/forced admission evidence for S4."""
from pathlib import Path
import ast
import hashlib
import json
import re

H=Path(__file__).parent;R=H.resolve().parents[3]
m=json.loads((H/'manifest.json').read_text());source=R/m['source']
assert hashlib.sha256(source.read_bytes()).hexdigest()==m['sha256']
screens={}
for label in ('normal','forced'):
    d=json.loads((H/(label+'_local.json')).read_text())
    assert d['sha256']==m['sha256'] and d['prepare_s']<300
    assert all(x['ok'] for k in ('tests','shapes','static') for x in d[k])
    assert [len(d[k]) for k in ('tests','shapes','static')]==[13,12,21]
    checks={x['name']:x['ok'] for x in d['static']}
    assert all(checks[n] for n in ('S4_NT1_CUDA','S4_D32_CUDA','RC_CUDA'))
    phases=re.search(r'BEST_PHASES seconds=(.*)',(H/(label+'.log')).read_text())
    screens[label]=dict(sha256=d['sha256'],prepare_s=d['prepare_s'],tests=13,graphs=12,static=21,
                        phases_s=ast.literal_eval(phases[1]),cc=d['cc'],gpu=d['gpu'])
f=json.loads((H/'forced_dispatch_local.json').read_text());assert f['sha256']==m['sha256']
calls=f['native_sm120'];assert [len(calls[n]['full_layer_rows']) for n in calls]==[8,2,2]
for row in calls.values():assert all(v['captured']>0 and v['eager']>0 for v in row['full_layer_rows'].values())
gates=f['forced']['gates'];selected={}
for tokens,name in ((1,'nt1_lop3'),(8,'nt1_lop3'),(16,'nt1_lop3'),(32,'nt1_lop3'),(128,'d32g_m128'),(256,'recoded_c1g2112')):
    key=str(('moe',tokens,name));gate=gates[key]
    assert gate['fp32']<=.01 and gate['worst_row']<=.05
    selected[key]=gate
result=dict(manifest=m,screens=screens,source_audit=json.loads((H/'audit.json').read_text()),
            forced_native_calls=calls,forced_gates=selected,forced_plans=f['forced']['plans'],
            unverified=['whole-layer H200 composition','cold H200 prepare','new judge seeds'])
(H/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(screens=screens,expert_gates=selected,native_calls=calls),indent=2))
