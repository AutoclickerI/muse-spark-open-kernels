"""Exact-SHA component receipts and public aggregate gaps; no invented leader shapes."""
import json,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
manifest=json.loads((H/'manifest.json').read_text());ledger=[json.loads(l) for l in (R/'runs/ledger.jsonl').read_text().rsplit('\n',1)[0].splitlines()]
rows=[];pending=[]
for c in manifest['candidates']:
    hits=[x for x in ledger if x['sha256']==c['sha256']]
    if not hits:pending.append(c['role']);continue
    entry=hits[-1];receipt=R/Path(entry['source']).with_suffix('.json');d=json.loads(receipt.read_text());r=d.get('result') or {}
    board=json.loads((H/('leader_'+c['role']+'.json')).read_text());leader=board['rows'][0]
    records=r.get('records',[]);tests=[a for a in records if a.get('event')=='test'];bench=[a for a in records if a.get('event')=='benchmark']
    expected={'moe':(5,6),'attention':(16,12),'projection':(9,30)}[c['role']]
    valid=bool(r.get('passed')) and (len(tests),len(bench))==expected and all(a.get('status')=='pass' for a in tests+bench) and not r.get('flags')
    record=dict(role=c['role'],job=entry['job'],receipt=str(receipt.relative_to(R)),sha256=c['sha256'],status=entry['status'],passed=valid,
        error=r.get('error'),tests=len(tests),shapes=len(bench),aa=None,aa_reason='official component receipt has SGLang paired controls; no independent A:A arm',
        leader_team=leader['team'],leader_job=leader['job'],leader_speedup=leader['speedup'],leader_score_us=leader['value'],leader_per_shape_available=False,
        board_snapshot='problems/muse_decode_layer/dev/p12_components/leader_'+c['role']+'.json')
    if valid:
        record.update(our_speedup=r['speedup'],our_score_us=r['score_us'],speedup_ci=r['speedup_ci'],
            time_ratio_to_component_leader=r['score_us']/leader['value'],normalized_time_ratio=leader['speedup']/r['speedup'],
            aggregate_excess_us=r['score_us']-leader['value'])
    record['our_shapes']=[dict(case=a['case'],us=a['median_ns']/1000,baseline_us=a['baseline_ns']/1000,speedup=a['speedup'],
        runs=a['runs'],ratio_min=min(a.get('ratios',[a['speedup']])),ratio_max=max(a.get('ratios',[a['speedup']])),status=a['status']) for a in bench]
    rows.append(record)
rows.sort(key=lambda a:a.get('time_ratio_to_component_leader',0),reverse=True)
out=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_sha256=manifest['base_sha256'],
    per_shape_limitation='leaderboard JSON contains aggregate value/speedup only; jobs endpoint is owner-only (upstream/llms.txt:78)',pending=pending,ranking=rows)
(H/'leader_gap.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r['role'],r['job'],'passed',r['passed'],'gap',r.get('time_ratio_to_component_leader'))
print('pending',pending)
