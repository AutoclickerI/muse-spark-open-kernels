"""Summarize both successful S3 screens with source and dispatch evidence."""
from pathlib import Path
import ast
import hashlib
import json
import re

H = Path(__file__).parent
R = H.resolve().parents[3]
m = json.loads((H / 'manifest.json').read_text())
source = R / m['source']
assert hashlib.sha256(source.read_bytes()).hexdigest() == m['sha256']
rows = {}
for label in ('normal', 'forced'):
    d = json.loads((H / (label + '_local.json')).read_text())
    assert d['sha256'] == m['sha256'], label
    assert d['prepare_s'] < 300, (label, d['prepare_s'])
    assert all(x['ok'] for k in ('tests', 'shapes', 'static') for x in d[k]), label
    assert (len(d['tests']), len(d['shapes']), len(d['static'])) == (13, 12, 19)
    phases = re.search(r'BEST_PHASES seconds=(.*)', (H / (label + '.log')).read_text())
    rows[label] = dict(sha256=d['sha256'], prepare_s=d['prepare_s'], tests=13, graphs=12, static=19,
                       phases_s=ast.literal_eval(phases[1]), cc=d['cc'], gpu=d['gpu'])
f = json.loads((H / 'forced_dispatch_local.json').read_text())
assert f['sha256'] == m['sha256']
calls = f['recoded_native_sm120']
assert set(calls['full_layer_rows']) == {'(32, 8, 2048)', '(32, 8, 0)'}
assert all(row['captured'] > 0 and row['eager'] > 0 for row in calls['full_layer_rows'].values())
gate = f['forced']['gates']["('moe', 256, 'recoded_c1g2112')"]
assert gate['encoded'] and (gate['which'], gate['grid']) == (1, 2112)
assert gate['fp32'] <= .01 and gate['worst_row'] <= .05
result = dict(manifest=m, screens=rows, source_audit=json.loads((H / 'audit.json').read_text()),
              forced_recoded_calls=calls, forced_gates=f['forced']['gates'],
              forced_plans=f['forced']['plans'], sent=False,
              unverified=['whole-layer H200 composition', 'cold H200 prepare', 'new judge seeds'])
(H / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(screens=rows, forced_recoded_calls=calls, m256_gate=gate, sent=False), indent=2))
