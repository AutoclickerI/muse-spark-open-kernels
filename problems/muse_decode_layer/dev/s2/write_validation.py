"""Save a compact source-pinned S2 handoff from both successful local screens."""
from pathlib import Path
import hashlib,json,re
H=Path(__file__).parent;R=H.resolve().parents[3]
m=json.loads((H/'manifest.json').read_text());source=R/m['source']
assert hashlib.sha256(source.read_bytes()).hexdigest()==m['sha256']
rows={}
for label in ('normal','forced'):
    d=json.loads((H/(label+'_local.json')).read_text())
    assert d['sha256']==m['sha256'],label
    assert d['prepare_s']<300,(label,d['prepare_s'])
    assert all(x['ok'] for k in ('tests','shapes','static') for x in d[k]),label
    assert (len(d['tests']),len(d['shapes']),len(d['static']))==(13,12,19)
    log=(H/(label+'.log')).read_text()
    phases=re.search(r'BEST_PHASES seconds=(.*)',log)
    import ast
    rows[label]=dict(sha256=d['sha256'],prepare_s=d['prepare_s'],tests=13,graphs=12,static=19,
                     phases_s=ast.literal_eval(phases[1]),cc=d['cc'],gpu=d['gpu'])
f=json.loads((H/'forced_dispatch_local.json').read_text())
assert f['sha256']==m['sha256']
assert set(f['w2fix_native_sm120']['full_layer_rows'])=={'(32, 8, 2048)','(32, 8, 0)'}
assert all(row['captured']>0 for row in f['w2fix_native_sm120']['full_layer_rows'].values())
result=dict(manifest=m,screens=rows,source_audit=json.loads((H/'audit.json').read_text()),
            forced_w2fix_calls=f['w2fix_native_sm120'],
            forced_gates=f['forced']['gates'],forced_plans=f['forced']['plans'],
            sent=False,unverified=['whole-layer H200 composition','cold H200 prepare','new judge seeds'])
cold=json.loads((H/'forced_cold_attempt_local.json').read_text())
assert cold['sha256']==m['sha256'] and cold['prepare_s']<300
assert all(x['ok'] for k in ('tests','shapes','static') for x in cold[k])
result['cold_attention_attempt']=dict(prepare_s=cold['prepare_s'],checks='13/12/19 pass',coverage_driver='initial Python-counter assertion invalid; corrected driver and full screen rerun')
(H/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ('screens','forced_w2fix_calls','sent')},indent=2))
