"""All-seed deletion sizing using G0's paired-log interval, never promotion."""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets
from promotion import interval
from sample_codec import unpack_samples

def analyze(path):
    receipt=json.loads(path.read_text());records=read_packets(receipt['result']['log_tail']);assert len(records)==1
    p=unpack_samples(records[0]['packet']);m=json.loads((H/'manifest.json').read_text())
    assert p['kind']=='P5_FINISH_DELETION_ORACLE' and p['complete'] and not p.get('transport_errors')
    assert p['base_sha256']==m['base_sha256'] and p['seed_ids']==m['seeds'] and p['source_hashes']==m['sources']
    assert p['measurement_protocol']==m['measurement_protocol'] and p['seeds']==p['seed_ids']==m['seeds']
    assert p['identities']==dict(base_sha256=m['base_sha256'],sources=m['sources'])
    assert records[0]['source_sha256']==m['payload_sha256'][:16]
    seeds=p['seed_ids'];n=len(seeds);rounds=m['measurement_protocol']['rounds']
    assert len(p['paired'])==12 and p['checks']['BASE']==p['checks']['AA']==12*n
    shapes={};means={}
    for row in p['paired']:
        assert len(row['round_ns'])==n and len(row['telemetry'])==n
        values=row['round_ns'];assert all(len(x)==3 and all(len(y)==rounds and all(v>0 for v in y) for y in x) for x in values)
        avg=[[statistics.mean(x) for x in seed] for seed in values];means[row['label']]=avg
        rr=[v[0]/v[2] for v in avg];aa=[v[1]/v[0] for v in avg]
        shapes[row['label']]=dict(R=interval(rr,aa),AA=interval(aa),BASE_us=statistics.mean(v[0] for v in avg)/1000,DEL_us=statistics.mean(v[2] for v in avg)/1000,DEL_TASK_fixture=row['DEL_fixture_TASK'])
    loads={}
    for b,l in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        label=('decode' if l==1 else 'verify')+' '+str(b);prefix=('d' if l==1 else 'v')+str(b)
        s,f=means[prefix+'S'],means[prefix+'F'];rr={k:[] for k in ('DEL','DEL_S','DEL_F')};aa=[]
        for sv,fv in zip(s,f):
            old=46*sv[0]+16*fv[0];aa.append((46*sv[1]+16*fv[1])/old)
            rr['DEL'].append(old/(46*sv[2]+16*fv[2]));rr['DEL_S'].append(old/(46*sv[2]+16*fv[0]));rr['DEL_F'].append(old/(46*sv[0]+16*fv[2]))
        loads[label]=dict(**{k:interval(v,aa) for k,v in rr.items()},AA=interval(aa),per_seed=rr,AA_per_seed=aa)
    six={};aa=[math.exp(statistics.mean(math.log(x['AA_per_seed'][i]) for x in loads.values())) for i in range(n)]
    for arm in ('DEL','DEL_S','DEL_F'):
        rr=[math.exp(statistics.mean(math.log(x['per_seed'][arm][i]) for x in loads.values())) for i in range(n)]
        six[arm]=interval(rr,aa)
    six['AA']=interval(aa)
    half=[math.exp(statistics.mean(math.log(2*x['per_seed']['DEL'][i]/(1+x['per_seed']['DEL'][i])) for x in loads.values())) for i in range(n)]
    half_R=math.exp(statistics.mean(map(math.log,half)))
    build=six['DEL']['CI95'][0]>=1.015
    out=dict(job=receipt['id'],receipt=str(path),base=p['base_sha256'],payload_sha256=m['payload_sha256'],scope=p['scope'],checks=p['checks'],seed_ids=seeds,loads=loads,six_load=six,shapes=shapes,half_deletion_time_saving_model=dict(R=half_R,scope='Illustration only: recover half of each measured load time difference at no added cost. Not a measured implementation or a bound.'),build_consumer_fusion=build,decision_threshold=1.015,decision='Build consumer prologue fusion; size FA3 hook versus native replacement.' if build else 'Below prescribed lower-bound threshold: park consumer-side finish family by this oracle.',elapsed_s=p['elapsed_s'],telemetry_error=p['telemetry_error'],attention_selections=p['attention_selections'],limitations=['Altered-work oracle; deliberately prepares timed-input values outside graph. Never a valid candidate or G0 promotion.','Per-path results combine measured changed path with the unchanged BASE path; no separately timed duplicate calls.','New binary and ready-data transport may change scheduling/resource use; S19 partition choices are pinned.'])
    (H/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    print('ORACLE',receipt['id'],'TASK',p['checks'],'telemetry_error',p['telemetry_error'])
    for label,row in loads.items():print(label,{k:(row[k]['R'],row[k]['CI95']) for k in ('DEL','DEL_S','DEL_F','AA')})
    print('SIX',six);print(out['decision']);return out
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('receipt',type=Path);analyze(a.parse_args().receipt)
