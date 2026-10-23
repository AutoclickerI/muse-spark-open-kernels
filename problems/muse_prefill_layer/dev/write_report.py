"""Render receipt state; lane prose is owned by FINDINGS.md and FOCUS.md."""
import datetime,hashlib,json
from pathlib import Path
from collections import defaultdict
P=Path(__file__).resolve().parents[1];ROOT=P.parents[1]
rows=[]
for f in sorted((ROOT/'runs/muse_prefill_layer').glob('*.json')):
    d=json.loads(f.read_text());r=d.get('result',{});loads=defaultdict(lambda:[0.,0.])
    for event in r.get('records',[]):
        if event.get('event')=='benchmark' and event.get('status')=='pass':
            c=event['case'];loads[c['load']][0]+=event['baseline_ns']*c['weight'];loads[c['load']][1]+=event['median_ns']*c['weight']
    rows.append(dict(job=d['id'],mode=d['mode'],passed=r.get('passed'),speedup=r.get('speedup'),
                     loads={k:a/b for k,(a,b) in loads.items()},error=r.get('error'),receipt=str(f.relative_to(ROOT))))
focus=P/'FOCUS.md'
state=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           focus=focus.read_text(),submission_sha256=hashlib.sha256((P/'submission.py').read_bytes()).hexdigest(),receipts=rows)
(P/'dev/receipt_state.json').write_text(json.dumps(state,indent=2))
print('Prefill receipt state refreshed; current work is in FOCUS.md')
