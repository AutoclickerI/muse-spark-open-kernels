"""Size P13 deletions with the unchanged promotion.py paired-log interval."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]
sys.path[:0]=[str(H),str(R/'tools')]
from codec import unpack_samples
from lab_packet import read_packets
from promotion import interval


def analyze(path):
    receipt=json.loads(path.read_text());records=read_packets(receipt['result']['log_tail'])
    assert len(records)==1
    assert records[0]['packet'].get('complete'),records[0]['packet']
    p=unpack_samples(records[0]['packet']);m=json.loads((H/'manifest.json').read_text())
    assert p['kind']=='P13_BOUNDARY_ROUTE_ORACLE' and p['complete'] and not p.get('transport_errors')
    assert p['identities']==dict(base_sha256=m['base_sha256']) and p['base_sha256']==m['base_sha256']
    assert p['measurement_protocol']==m['measurement_protocol'] and p['seeds']==p['seed_ids']==m['seeds']
    assert records[0]['source_sha256']==m['payload_sha256'][:16] and p['arms']==m['arms']
    n=len(p['seeds']);rounds=m['measurement_protocol']['rounds'];arms=p['arms']
    assert len(p['paired'])==12
    assert all(v==dict(pass_count=12*n,fail_count=0) for v in p['checks'].values())
    means={};shapes={}
    for row in p['paired']:
        values=row['round_ns']
        assert len(values)==len(row['telemetry'])==len(row['fixture_TASK'])==n
        assert all(len(seed)==len(arms) and all(len(r)==rounds and all(v>0 for v in r) for r in seed) for seed in values)
        assert all(len(t)==rounds//2+1 for t in row['telemetry'])
        assert len(row['local_start'])==n and all(v%32==0 and 0<=v<256 for v in row['local_start'])
        avg=[{arm:statistics.mean(v) for arm,v in zip(arms,seed)} for seed in values]
        means[row['label']]=avg
        aa=[v['BASE']/v['AA'] for v in avg]
        shapes[row['label']]={arm:interval([v['BASE']/v[arm] for v in avg],aa) for arm in arms[1:]}
        shapes[row['label']]['us']={arm:statistics.mean(v[arm] for v in avg)/1000 for arm in arms}
    loads={}
    for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        label=('decode' if l==1 else 'verify')+' '+str(b);prefix=('d' if l==1 else 'v')+str(b)
        values=[{arm:46*s[arm]+16*f[arm] for arm in arms} for s,f in zip(means[prefix+'S'],means[prefix+'F'])]
        rr={arm:[v['BASE']/v[arm] for v in values] for arm in arms[1:]}
        loads[label]=dict(ratios={arm:interval(v,rr['AA']) for arm,v in rr.items()},per_seed=rr,
                          weighted_layer_us={arm:statistics.mean(v[arm] for v in values)/62000 for arm in arms})
    six_samples={arm:[math.exp(statistics.mean(math.log(row['per_seed'][arm][i]) for row in loads.values())) for i in range(n)] for arm in arms[1:]}
    six={arm:interval(rr,six_samples['AA']) for arm,rr in six_samples.items()}
    best=max(('ROUTE','BOUND','BOTH'),key=lambda arm:six[arm]['CI95'][0])
    build=six[best]['CI95'][0]>=1.01
    out=dict(job=receipt['id'],receipt=str(path.relative_to(R)),base_sha256=p['base_sha256'],
             payload_sha256=m['payload_sha256'],scope=p['scope'],checks=p['checks'],seed_ids=p['seed_ids'],
             measurement_protocol=p['measurement_protocol'],loads=loads,six_load=six,shapes=shapes,
             best_arm=best,build_fusion=build,decision_threshold=1.01,
             AA_reference_sensitivity={arm:interval([math.exp(statistics.mean(math.log(row['per_seed'][arm][i]/row['per_seed']['AA'][i]) for row in loads.values())) for i in range(n)]) for arm in ('ROUTE','BOUND','BOTH')},
             decision='Build fusion against the measured deletion.' if build else 'Park P13 boundary/route family: best six-load lower bound below 1.01.',
             elapsed_s=p['elapsed_s'],telemetry_error=p['telemetry_error'],
             local_start={row['label']:row['local_start'] for row in p['paired']},
             limitations=['Altered-work oracle, not an admissible candidate or promotion.',
                          'Deletion changes graph dependency/resource overlap; the causal ceiling is local to this fixture protocol.',
                          'Only separate route/metadata kernels are deleted; metadata inside expert compute remains timed.'])
    (H/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    (H/'out/native_packet.json').write_text(json.dumps(p)+'\n')
    print('RECEIPT',receipt['id'],'checks',p['checks'])
    for label,row in loads.items():print(label,{arm:round(v['R'],6) for arm,v in row['ratios'].items()})
    print('SIX',six);print(out['decision'])
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path)
    analyze(parser.parse_args().receipt.resolve())
