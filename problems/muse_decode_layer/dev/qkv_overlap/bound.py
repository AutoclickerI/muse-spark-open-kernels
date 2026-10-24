"""Offline two-KV-group flow-shop bound from source-pinned H200 receipts."""
from pathlib import Path
import json,hashlib,re,math
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best7_reset.c0e74bedf7b74be5.json'
proof=ROOT/'problems/muse_bf16_proj/profiles/submission.7fb3e93056d34999.json'
focus=ROOT/'problems/muse_decode_layer/dev/best7/sensitivity.json'
d=json.loads(p.read_text());sensitivity=json.loads(focus.read_text());standalone=json.loads(proof.read_text())
pattern=r'^(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)\s*$'
def kernels(report):
    return [dict(name=m[1],us=float(m[2]),calls=int(m[3])) for line in report.splitlines() if (m:=re.match(pattern,line))]
qkv_proof={x['case']['m']:kernels(x['report']) for x in standalone['profiles'] if x['case']['layer']=='qkv_gate' and x['case']['m'] in (1,16,32)}
rows=[]
for profile in d['profiles']:
    c=profile['case']
    if c.get('q_len',1)!=1:continue
    ks=kernels(profile['report']);m=c['batch']
    # The same main/reduce names appear in isolated qkv profiles at each M;
    # this resolves the whole-layer nvjet role grouping without floor allocation.
    names={k['name'] for k in qkv_proof[m]}
    chosen=[k for k in ks if k['name'] in names]
    assert len(chosen)==len(names) and len(names)==2,(m,chosen,qkv_proof[m])
    q=sum(k['us'] for k in chosen)
    a=sum(k['us'] for k in ks if 'attn_main' in k['name'] or 'attn_combine' in k['name'])
    saving=.5*min(q,a)
    layer=dict(case=c,qkv_us=q,attention_us=a,ideal_saved_us=saving,
       ideal_serial_prefix_us=q+a,ideal_two_group_prefix_us=.5*(q+a+max(q,a)),qkv_kernels=chosen)
    assert math.isclose(layer['ideal_serial_prefix_us']-layer['ideal_two_group_prefix_us'],saving)
    rows.append(layer)
assert len(rows)==6
# Use the FOCUS sensitivity's exact six-load denominator (best6 benchmark medians).
loads={name:v['us'] for name,v in sensitivity['loads'].items()}
# Cross-check the profile's own complete benchmark times as a second denominator.
reset_loads={}
for b in d['benchmarks']:
    c=b['case'];reset_loads[c['load']]=reset_loads.get(c['load'],0.)+c['weight']*b['median_ns']/1000
savings={}
for r in rows:savings[r['case']['load']]=savings.get(r['case']['load'],0.)+r['case']['weight']*r['ideal_saved_us']
score=lambda times,save:100*math.expm1(sum(math.log(times[n]/(times[n]-v)) for n,v in save.items())/6)
byload={name:dict(saved_weighted_us=saved,focus_denominator_us=loads[name],
    isolated_overall_gain_pct=score(loads,{name:saved}),reset_overall_gain_pct=score(reset_loads,{name:saved})) for name,saved in savings.items()}
result=dict(kind='optimistic bound, not measured overlap',kv_groups=2,rows=rows,by_load=byload,
    all_decode_overall_gain_pct=score(loads,savings),reset_all_decode_overall_gain_pct=score(reset_loads,savings),
    formula='saved per layer=(1-1/2)*min(Q,A); score=product_load[T/(T-46*dS-16*dF)]**(1/6)-1',
    assumptions=['Two equal KV-head groups; all Q/K/V/gate rows for a group arrive at Q/2 or Q.',
      'Both pipeline stages retain their measured aggregate throughput during overlap, with zero added launches, synchronization, or head-group tails.',
      'Attention group work is sequential on the attention resource; qkv and attention resources are unrealistically independent.',
      'Profile GPU durations are activity envelopes, possibly including PDL waits; the result is a screening bound, not a forecast.'],
    sources={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in (p,proof,focus)},
    build_threshold_pct=1.,threshold_passed=score(loads,savings)>1.)
(HERE/'bound.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# QKV / attention KV-group overlap ceiling','',
    'Two KV heads give two independent groups of eight Q/gate heads. For equal groups,',
    'an ideal two-stage pipeline saves `min(Q,A)/2`. This assumes full standalone',
    'throughput while overlapping and zero extra launches, reductions or synchronization.',
    'QKV main+reduce names are matched to isolated qkv H200 profiles; attention includes',
    'its combine where present. Denominators are the FOCUS sensitivity benchmark medians.','',
    '| Decode load | QKV S/F us | Attention S/F us | Ideal saved S/F us | Overall gain from this load |',
    '|---|---:|---:|---:|---:|']
for m in (1,16,32):
    sr=next(r for r in rows if r['case']['batch']==m and r['case']['layer']=='sliding');fr=next(r for r in rows if r['case']['batch']==m and r['case']['layer']=='full')
    pair=lambda key:f'{sr[key]:.2f}/{fr[key]:.2f}'
    lines.append(f'| {m} | {pair("qkv_us")} | {pair("attention_us")} | {pair("ideal_saved_us")} | {byload[f"decode {m}"]["isolated_overall_gain_pct"]:.3f}% |')
lines+=['',f'All three decode loads together: **{result["all_decode_overall_gain_pct"]:.3f}% overall**.',
    f'Using best7_reset\'s own benchmark denominators instead: {result["reset_all_decode_overall_gain_pct"]:.3f}%.',
    'This clears the1% screening threshold only in aggregate. At 50% realization, the',
    f'overall gain is {score(loads,{k:v*.5 for k,v in savings.items()}):.3f}%.',
    'The bound gives no GPU-resource availability or scheduling guarantee. M1/16/32',
    'qkv and attention compete for the same SMs and memory; launch cost is not divisible',
    'by head count. Splitting the current Lt split-K/reduction schedule can add cost.',
    'No overlap kernel has been built or sent; a future design must stream each',
    'group\'s complete Q/K/V/gates and preserve per-head QK RMS and cache publication.',
    'Source hashes and exact inputs are in `bound.json`.']
(HERE/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
