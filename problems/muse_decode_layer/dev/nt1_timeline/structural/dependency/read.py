"""Verify the exact submitted packet, then report paired stage and output-path costs."""
import argparse, ast, base64, hashlib, json, lzma, statistics, sys
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[5]
sys.path.insert(0, str(R / 'tools'))
import lab_packet

def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values, critical=2.131449545559323):
    n = len(values)
    mean = statistics.mean(values)
    se = statistics.stdev(values) / n**.5 if n > 1 else 0.
    return dict(mean=mean, SE=se, CI95=[mean-critical*se, mean+critical*se], n=n)

p = argparse.ArgumentParser()
p.add_argument('receipt', type=Path)
a = p.parse_args()
host = read(a.receipt)
q = read(D/'queue.json')
payload = read(D/'payload.json')
cells = read(D/'cells.json')
manifest = read(D/'manifest.json')
counts = read(D/'counts.json')['cells']
assert host['id'] == q['job']
source = (R/q['source']).read_text()
assert hashlib.sha256(source.encode()).hexdigest() == q['sha256']
entries = next(ast.literal_eval(n.value) for n in ast.parse(source).body
               if isinstance(n, ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets))
name, short_sha, body = next(v for v in entries if v[1] == payload['sha256'][:16])
assert hashlib.sha256(body.encode()).hexdigest() == payload['sha256']
blob = next(n.args[0].value for n in ast.walk(ast.parse(body))
            if isinstance(n,ast.Call) and getattr(n.func,'attr','') == 'b85decode')
expanded = lzma.decompress(base64.b85decode(blob)).decode()
assert hashlib.sha256(expanded.encode()).hexdigest() == payload['expanded_sha256']
for spec in cells.values():
    text = (R/spec['cell']).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == spec['cell_sha256'] and text in expanded
records = lab_packet.read_packets(host['result'].get('log_tail',''))
record = next(v for v in records if v['source_sha256'] == short_sha)
r = record['packet']
assert r['kind'] == 'P3A_DEPENDENCY' and r['complete'] and not r.get('transport_errors')
assert r['identities'] == payload['identities']
assert not r['checks']['failures'], r['checks']
assert r['seeds']['oracle_complete'] == r['seeds']['requested'] == 16
assert len(r['oracle']) == len(r['AA']) == 48
for arm, gate in r['checks']['gates'].items():
    assert gate[0] <= .015 and gate[1] <= .075, (arm, gate)
resources = []
assert len(r['resources']) == 27
for m, arm, info in r['resources']:
    if arm in ('BASE','NOWAIT'):
        c = counts[arm+('1' if m == 8 else '2')]['parts']['69']
    else:
        S = 64 if arm == 'A64' else 128
        mode = 2 if arm.startswith('NORED') else 1 if arm.startswith('P') else 0
        c = counts['DEEP' if arm in ('A128D','P128D','NOREDD') else 'SLICE']['parts'][f'{S}_{mode}']
    offline = next(v for v in c['rows'] if v['M'] == m)
    expected = [offline[k] for k in ('resident','registers','stack','dynamic','static')]
    assert info == expected, (m,arm,info,expected)
    resources.append(dict(M=m,arm=arm,runtime=info,offline_match=True))
summary, oracle, aa_summary = [], [], []
for m in (8,16,32):
    aa = [v for v in r['AA'] if v[1] == m]
    old = [v for v in r['oracle'] if v[1] == m]
    for rows in (aa,old): assert sorted(v[0] for v in rows) == list(range(16))
    aa_s = stats([v[4]/1e6 for v in aa])
    aa_s.update(M=m, mean_abs_us=statistics.mean(abs(v[3]-v[2])/1000 for v in aa),
                max_abs_us=max(abs(v[3]-v[2])/1000 for v in aa))
    aa_summary.append(aa_s)
    o = stats([v[4]/1e6 for v in old])
    o.update(M=m, ratio=o['mean'], seeds=16, seed_ratios=[v[4]/1e6 for v in old],
             baseline_us=statistics.mean(v[2]/1000 for v in old),
             oracle_us=statistics.mean(v[3]/1000 for v in old),
             saved_us=stats([(v[2]-v[3])/1000 for v in old]),
             point_nominee=False, eligibility='Timing only; deliberately incorrect output')
    oracle.append(o)
    for arm in manifest:
        rows = [v for v in r['paired'] if v[1] == m and r['arm_names'][v[2]] == arm]
        if not r['material']:
            assert not rows
            continue
        assert sorted(v[0] for v in rows) == list(range(16))
        ratios = [v[3]/1e6 for v in rows]
        s = stats(ratios)
        summary.append(dict(M=m,arm=arm,control='S19',ratio=s['mean'],SE=s['SE'],CI95=s['CI95'],
                            seeds=16,seed_indices=list(range(16)),seed_ratios=ratios,
                            point_nominee=s['mean'] < 1, AA_mean_ratio=aa_s['mean'],AA_SE=aa_s['SE'],
                            AA_mean_abs_us=aa_s['mean_abs_us'],AA_max_abs_us=aa_s['max_abs_us']))
assert r['material'] == any(v['ratio'] < .95 for v in oracle if v['M'] in (16,32))
assert r['seeds']['build_complete'] == (16 if r['material'] else 0)
assert len(r['paired']) == (240 if r['material'] else 0)
profile_summary, output_cost, work = [], [], []
for m in (8,16,32):
    arms = r['arm_names'] if r['material'] else ['BASE','NOWAIT']
    by_arm = {}
    for arm in arms:
        rows = [v for v in r['profiles'] if v[1] == m and r['arm_names'][v[2]] == arm]
        assert sorted(v[0] for v in rows) == list(range(4))
        rows.sort()
        by_arm[arm] = rows
        profile_summary.append(dict(M=m,arm=arm,unit='us',first_seeds=4,
                                     **{key:stats([v[i+3]/1000 for v in rows],3.182446305284263)
                                        for i,key in enumerate(r['profile_columns'])}))
    if not r['material']: continue
    costs = [v for v in r['cost_counts'] if v[1] == m]
    assert sorted(v[0] for v in costs) == list(range(16))
    for arm,spec in manifest.items():
        ns = 4096//spec['S']
        normal = by_arm[arm]
        no = by_arm['NOREDD' if arm.endswith('D') else 'NORED']
        # NORED has S128; do not conflate the different S64 issue schedule with reduction cost.
        if ns == 32:
            output_cost.append(dict(M=m,arm=arm,unit='us',first_seeds=4,
                main_minus_NORED=stats([(v[3]-w[3])/1000 for v,w in zip(normal,no)],3.182446305284263),
                zero_norm_combine=stats([sum(v[4:])/1000 for v in normal],3.182446305284263),
                note='Net output-path contrast,not exclusive atomic latency;NORED107vs108regs,same2CTA;profile node sums omit launch gaps'))
        work.append(dict(M=m,arm=arm,slices=ns,mean_local_routes=statistics.mean(v[2] for v in costs),
                         mean_token_blocks=statistics.mean(v[3] for v in costs),
                         mean_useful_units=statistics.mean(ns*v[3] for v in costs),
                         mean_fp32_reductions=statistics.mean(v[2]*4096*ns for v in costs) if spec['mode']==0 else 0,
                         mean_partial_write_read_bytes=statistics.mean(8*v[2]*4096*ns for v in costs) if spec['mode']==1 else 0,
                         zero_bytes=4*m*8*4096 if spec['mode']==0 else 0,
                         total_TMA_copy_ratio_vs_BASE=2 if ns==32 else 10/3,
                         W2_TMA_copy_ratio_vs_BASE=4 if ns==32 else 8,
                         weight_byte_ratio_vs_BASE=1,
                         note='Logical counts,not measured physical HBM/L2 traffic'))
out = dict(job=q['job'],measured=True,host_passed=host['result'].get('passed'),invalid=[],
           source_audit=dict(verified=True,wrapper_sha256=q['sha256'],payload_sha256=payload['sha256'],
                             expanded_sha256=payload['expanded_sha256'],packet_checksum=record['checksum'],exact_cells=len(cells)),
           resources=resources,material=r['material'],oracle=oracle,AA=aa_summary,summary=summary,
           profile_summary=profile_summary,output_cost=output_cost,work=work,seed_level_record=r,
           promotion='All valid point winners to P4;held-out six-load R lower bound>1 confirms;A:A/CI descriptive')
(D/'results.json').write_text(json.dumps(out,indent=2)+'\n')
q.update(state='Complete16seeds/TASK/AA/packet/source/resources verified',
         receipt=str(a.receipt.resolve().relative_to(R)))
(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
for v in oracle: print('ORACLE',v['M'],round(v['ratio'],6),v['CI95'],'saved_us',v['saved_us']['mean'])
for v in summary: print(v['M'],v['arm'],round(v['ratio'],6),[round(t,6) for t in v['CI95']],v['point_nominee'],'AA',round(v['AA_mean_ratio'],6))
for v in output_cost: print('COST',v['M'],v['arm'],v['main_minus_NORED']['mean'],v['zero_norm_combine']['mean'])
