"""Evaluate each prospectively declared candidate using its own same-source A:A."""
import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
sys.path[:0]=[str(H.parent),str(R/'tools')]
from codec import unpack_samples
from lab_packet import read_packets
from promotion import evaluate,interval

def report(path,kind):
    receipt=json.loads(path.read_text());m=json.loads((H/(kind+'_lab_manifest.json')).read_text())
    records=read_packets(receipt['result']['log_tail']);assert len(records)==1
    envelope=records[0]['packet'];assert envelope.get('complete') and not envelope.get('transport_errors'),envelope
    assert records[0]['source_sha256']==m['payload_sha256'][:16]
    p=unpack_samples(envelope);assert p['kind']=='P13_DISTRIBUTED_'+kind.upper()
    assert p['identities']==m['identities'] and p['base_sha256']==m['base_sha256']
    assert p['seeds']==p['seed_ids']==m['confirmation_seeds'] and p['measurement_protocol']==m['measurement_protocol']
    n=len(p['seed_ids']);rounds=m['measurement_protocol']['rounds'];names=p['arms'];means={}
    assert names==['INC','A','A2','B','B2'] and len(p['paired'])==12
    for row in p['paired']:
        assert len(row['round_ns'])==len(row['telemetry'])==len(row['local_start'])==n
        assert all(len(seed)==rounds//2+1 and all(len(v)==5 for v in seed) for seed in row['telemetry'])
        assert all(v%32==0 and 0<=v<256 for v in row['local_start'])
        assert all(len(seed)==5 and all(len(v)==rounds and all(x>0 for x in v) for v in seed) for seed in row['round_ns'])
        means[row['label']]=[{name:statistics.mean(v) for name,v in zip(names,seed)} for seed in row['round_ns']]
    loads={}
    for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        prefix=('d' if l==1 else 'v')+str(b);label=('decode' if l==1 else 'verify')+' '+str(b)
        loads[label]=[{name:46*s[name]+16*f[name] for name in names} for s,f in zip(means[prefix+'S'],means[prefix+'F'])]
    outputs={}
    for candidate,control in (('A','A2'),('B','B2')):
        mapping={'INC':'INC','A':candidate,'A2':control}
        ids={name:m['identities'][actual] for name,actual in mapping.items()}
        manifest=dict(sha256=ids['A'],base_sha256=ids['INC'],baseline_arm='INC',arms={'INC':dict(sha256=ids['INC'],rows=[])},
                      confirmation_seeds=m['confirmation_seeds'],selection_seeds=m['selection_seeds'],checker_sha256=m['checker_sha256'],
                      nearby_arms=['A','INC'],nearby_expected_checks=36,expected_TASK_checks=3*12*n,measurement_protocol=m['measurement_protocol'])
        statuses={}
        for name,actual in mapping.items():
            failures=[dict(scope='arm',arms=[name],reason=f['reason']) for f in p['failures'] if f['arm']==actual]
            statuses[name]=dict(sha256=ids[name],native=receipt['result'].get('gpu')=='H200',complete=True,seed_ids=p['seed_ids'],
                                task_checks=p['checks'][actual]['pass_count'],expected_task_checks=12*n,nearby_task_checks=p['nearby_checks'][actual],
                                nearby_expected_checks=18 if name in ('A','INC') else 0,failures=failures)
        raw=dict(host_sha256=ids['A'],checker_sha256=m['checker_sha256'],seed_ids=p['seed_ids'],seeds=n,complete=True,failures=[],
                 loads={label:dict(samples=[{name:r[actual] for name,actual in mapping.items()} for r in samples]) for label,samples in loads.items()},
                 arm_status=statuses,measurement_protocol=m['measurement_protocol'],
                 timing_round_counts={label:{name:[rounds]*n for name in mapping} for label in means})
        value=evaluate(receipt,raw,manifest);score=value['score']
        aa=[math.exp(statistics.mean(math.log(samples[i][control]/samples[i][candidate]) for samples in loads.values())) for i in range(n)]
        lab=(score['measured'] and score['CI95'][0]>1 and p['prepare_s'][candidate]<=300 and all(not st['failures'] and st['nearby_task_checks']==st['nearby_expected_checks'] for st in statuses.values()))
        outputs[candidate]=dict(sha256=ids['A'],score=score,AA=interval(aa),lab_G0=lab,prepare_estimate_s=p['prepare_s'][candidate],
                               official_promotion=value['freeze_gate'],status=statuses,
                               weighted_layer_us={label:{name:statistics.mean(r[actual] for r in samples)/62000 for name,actual in mapping.items()} for label,samples in loads.items()})
    winners=[name for name,v in outputs.items() if v['lab_G0']]
    result=dict(job=receipt['id'],receipt=str(path.relative_to(R)),kind=kind,base_series=m['base_series'],base_sha256=m['base_sha256'],
                payload_sha256=m['payload_sha256'],candidates=outputs,lab_winners=winners,checks=p['checks'],nearby=p['nearby_checks'],failures=p['failures'],
                protocol=p['measurement_protocol'],telemetry_error=p['telemetry_error'],elapsed_s=p['elapsed_s'],
                scope='Native whole production graph vs pinned P4 source, all samples retained. Lab G0 is not official promotion; P4 confirms the exact composition on new disjoint seeds.')
    (H/(kind+'_result.json')).write_text(json.dumps(result,indent=2)+'\n')
    patches={}
    for name in winners:
        patch=H/'out'/(kind+'_'+name+'.inc');tail=patch.read_text()
        assert hashlib.sha256(tail.encode()).hexdigest()==m['arms'][name]['suffix_sha256']
        patches[name]=dict(operation='append newline then suffix to the exact expanded base',
                           base_sha256=m['base_sha256'],result_sha256=outputs[name]['sha256'],
                           suffix_sha256=m['arms'][name]['suffix_sha256'],suffix=tail)
    handoff=dict(**{k:result[k] for k in ('job','kind','base_series','base_sha256','lab_winners')},
                 source_files=['mechanisms.py','boundary.cu','boundary.py.inc'] if kind=='boundary' else ['mechanisms.py','route.py.inc','../fused.cu.inc'],
                 candidates={name:dict(sha256=v['sha256'],score=v['score'],prepare_estimate_s=v['prepare_estimate_s']) for name,v in outputs.items()},
                 exact_patches=patches,
                 action='P4 rebase qualifying exact suffix onto its newest reported S2x and confirm the composition on fresh seeds.' if winners else 'No qualifying native G0 winner in this step; close only these recipes.',
                 official_promotion=False)
    (H/(kind+'_HANDOFF.json')).write_text(json.dumps(handoff,indent=2)+'\n')
    print(kind,receipt['id'],'winners',winners)
    for name,v in outputs.items():print(name,v['score'].get('R'),v['score'].get('CI95'),{k:r['R'] for k,r in v['score'].get('loads',{}).items()})
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('kind',choices=('boundary','route'));parser.add_argument('receipt',type=Path);args=parser.parse_args();report(args.receipt.resolve(),args.kind)
