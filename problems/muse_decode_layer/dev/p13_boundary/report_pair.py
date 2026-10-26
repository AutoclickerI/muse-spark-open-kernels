"""Use tools/promotion.py without altering G0 or the official TASK gate."""
import argparse
import hashlib
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
from promotion import evaluate,interval


def main():
    parser=argparse.ArgumentParser();parser.add_argument('receipt',type=Path);a=parser.parse_args()
    path=a.receipt.resolve();receipt=json.loads(path.read_text());records=read_packets(receipt['result']['log_tail'])
    assert len(records)==1
    m=json.loads((H/'pair_manifest.json').read_text());packet=records[0]['packet']
    assert packet.get('complete') and not packet.get('transport_errors'),packet
    assert packet['kind']=='P13_FUSION_PAIR' and records[0]['source_sha256']==m['payload_sha256'][:16]
    assert packet['identities']==m['identities'] and packet['seeds']==packet['seed_ids']==m['confirmation_seeds']
    assert packet['measurement_protocol']==m['measurement_protocol']
    p=unpack_samples(packet);names=p['arms'];n=len(p['seed_ids']);rounds=m['measurement_protocol']['rounds']
    assert names==['INC','A','A2','SMALL','LARGE'] and len(p['paired'])==12
    means={};round_counts={}
    for row in p['paired']:
        assert len(row['round_ns'])==len(row['telemetry'])==n
        assert len(row['local_start'])==n and all(v%32==0 and 0<=v<256 for v in row['local_start'])
        assert all(len(seed)==len(names) and all(len(v)==rounds and all(x>0 for x in v) for v in seed) for seed in row['round_ns'])
        means[row['label']]=[{name:statistics.mean(v) for name,v in zip(names,seed)} for seed in row['round_ns']]
        round_counts[row['label']]={name:[rounds]*n for name in names}
    loads={}
    for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        prefix=('d' if l==1 else 'v')+str(b);label=('decode' if l==1 else 'verify')+' '+str(b)
        samples=[{name:46*s[name]+16*f[name] for name in names} for s,f in zip(means[prefix+'S'],means[prefix+'F'])]
        loads[label]=dict(samples=samples)
    statuses={}
    for name in names:
        c=p['checks'][name];expected=len(m['nearby_cases']) if name in m['nearby_arms'] else 0
        statuses[name]=dict(sha256=m['identities'][name],native=receipt['result'].get('gpu')=='H200',complete=True,
                            seed_ids=p['seed_ids'],task_checks=c['pass_count'],expected_task_checks=12*n,
                            nearby_task_checks=p['nearby_checks'][name],nearby_expected_checks=expected,
                            failures=[] if not c['fail_count'] else [dict(scope='arm',reason='TASK failure')])
    raw=dict(host_sha256=m['sha256'],checker_sha256=m['checker_sha256'],seed_ids=p['seed_ids'],seeds=n,
             complete=p['complete'],failures=[],loads=loads,arm_status=statuses,
             measurement_protocol=p['measurement_protocol'],timing_round_counts=round_counts,
             task_checks=sum(c['pass_count'] for c in p['checks'].values()),expected_task_checks=12*n*len(names),
             nearby_task_checks=sum(p['nearby_checks'].values()),nearby_expected_checks=m['nearby_expected_checks'])
    # An import lab cannot assert the official candidate TASK result. Leave
    # the promotion evaluator's independent official gate false, exactly as P4.
    result=evaluate(receipt,raw,m)
    lab_gate=(result['score']['measured'] and result['score']['CI95'][0]>1 and p['prepare_s']['A']<=300 and
              all(not statuses[name]['failures'] and statuses[name]['nearby_task_checks']==statuses[name]['nearby_expected_checks'] for name in ('INC','A','A2')))
    aa=[math.exp(statistics.mean(math.log(row['samples'][i]['A2']/row['samples'][i]['A']) for row in loads.values())) for i in range(n)]
    diagnostics={}
    for arm in ('A','SMALL','LARGE'):
        per_load={label:[row['INC']/row[arm] for row in values['samples']] for label,values in loads.items()}
        rr=[math.exp(statistics.mean(math.log(v[i]) for v in per_load.values())) for i in range(n)]
        diagnostics[arm]=dict(six=interval(rr,aa),loads={label:interval(v) for label,v in per_load.items()},
                              scope='Prespecified diagnostic vs S19; only combined A has an identical-source A:A in this batch.' if arm!='A' else 'Primary promotion.py comparison')
    summary={k:v for k,v in result.items() if k not in ('raw','arm_status')}
    summary.update(lab_G0=lab_gate,official_TASK_pass=False,official_promotion=False,checks=p['checks'],nearby_checks=p['nearby_checks'],
                   prepare_s=p['prepare_s'],prepare_gate=p['prepare_s']['A']<=300,protocol=p['measurement_protocol'],
                   telemetry_error=p['telemetry_error'],elapsed_s=p['elapsed_s'],diagnostics=diagnostics,
                   receipt=str(path.relative_to(R)),payload_sha256=m['payload_sha256'])
    summary['AA_control']=interval(aa)
    summary['weighted_layer_us']={label:{name:statistics.mean(row[name] for row in values['samples'])/62000 for name in names} for label,values in loads.items()}
    summary['local_start']={row['label']:row['local_start'] for row in p['paired']}
    (H/'fusion_result.json').write_text(json.dumps(summary,indent=2)+'\n')
    (H/'out/fusion_native_packet.json').write_text(json.dumps(p)+'\n')
    handoff=dict(job=receipt['id'],base_sha256=m['base_sha256'],candidate_sha256=m['sha256'],lab_G0=lab_gate,
                 score=result['score'],prepare_s=p['prepare_s'],official_TASK_pass=False,
                 action='P4 confirm the exact combined patch in the next composition.' if lab_gate else 'No combined G0 winner; retain separate-arm diagnostics, not a promotion.',
                 exact_patch_files=['build_fusion.py','fusion.py.inc','fused.cu.inc'],rebuild='python problems/muse_decode_layer/dev/p13_boundary/build_fusion.py',
                 diagnostics=diagnostics,oracle_job=m['oracle'])
    (H/'P4_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n')
    print('JOB',receipt['id'],'measured',result['score']['measured'],'lab_G0',lab_gate,'official promotion',result['freeze_gate'])
    print('COMBINED',result['score'].get('R'),result['score'].get('CI95'))
    print('CHECKS',p['checks'],p['nearby_checks'],'prepare',p['prepare_s'])
    for arm,v in diagnostics.items():print(arm,v['six'],{k:round(w['R'],6) for k,w in v['loads'].items()})


if __name__=='__main__':main()
