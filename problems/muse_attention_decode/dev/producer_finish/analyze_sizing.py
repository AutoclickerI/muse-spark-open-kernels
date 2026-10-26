"""Sizing evidence, not a fabricated fused-arm or six-load promotion result."""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
from scipy.stats import t
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets
def stats(xs):
    mean=statistics.mean(xs);se=statistics.stdev(xs)/math.sqrt(len(xs));q=float(t.ppf(.975,len(xs)-1))
    return dict(mean=mean,SE=se,CI95=[mean-q*se,mean+q*se],n=len(xs))
def analyze(path):
    receipt=json.loads(path.read_text());records=read_packets(receipt['result']['log_tail']);assert len(records)==1
    p=records[0]['packet'];assert p['complete'] and not p.get('transport_errors')
    assert p['identities']['base_sha256']=='62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a'
    assert p['identities']['finish_sha256']==hashlib.sha256((H/'finish.cu').read_bytes()).hexdigest()
    assert p['checks']==dict(layer_TASK=192,finish_TASK=96)
    shape={};loads=[]
    for r in p['paired']:
        n=len(r['seeds']);assert n==8 and r['TASK']==2*n and r['finish_TASK']==n
        v=r['times_ns'];c=r['critical'];label=f"{'d' if r['L']==1 else 'v'}{r['B']}{'S' if r['window'] else 'F'}"
        floor=1.3*r['B']*(2048+r['L']-1)*2*64*2/4.8e6 if r['window'] else None
        row=dict(B=r['B'],L=r['L'],window=r['window'],seeds=r['seeds'],layer_A_A=stats([a/b for a,b in zip(v['A2'],v['A'])]),layer_event_us=stats([x/1000 for x in v['A']]),attention_critical_us=stats([x['attn'] for x in c]),producer_critical_us=stats([x.get(r['abi']['role'],0) for x in c]),producer_ABI=r['abi'],
                 finish_probe_ratio=stats([a/b for a,b in zip(v['PF'],v['PE'])]),finish_minus_empty_us=stats([(a-b)/1000 for a,b in zip(v['PF'],v['PE'])]),finish_minus_producer_us=stats([(a-b)/1000 for a,b in zip(v['PF'],v['P'])]),finish_body_median_ns=stats([x[1] for x in r['finish_clocks_ns']]),finish_body_max_ns=stats([x[2] for x in r['finish_clocks_ns']]),producer_nodes=r['producer_nodes'],KV_target_us=floor,
                 remaining_sliding_gap_us=None if floor is None else stats([x['attn']-floor for x in c]))
        shape[label]=row
    for B,L in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
        s=next(r for r in p['paired'] if (r['B'],r['L'],r['window'])==(B,L,2048));f=next(r for r in p['paired'] if (r['B'],r['L'],r['window'])==(B,L,0))
        weighted=lambda arm:[46*a+16*b for a,b in zip(s['times_ns'][arm],f['times_ns'][arm])]
        old,new=weighted('A'),weighted('A2');fp,fe=weighted('PF'),weighted('PE')
        loads.append(dict(B=B,L=L,label=f"{'decode' if L==1 else 'verify'} {B}",seeds=s['seeds'],layer_A_A=stats([a/b for a,b in zip(new,old)]),finish_probe_ratio=stats([a/b for a,b in zip(fp,fe)]),weighted_finish_minus_empty_per_layer_us=stats([(a-b)/62/1000 for a,b in zip(fp,fe)])))
    # Size the four supplied preparation estimates; d1/v1 are explicitly
    # unchanged in this scenario, not silently assumed measured.
    prior=json.loads((H.parent/'setup_tree/result.json').read_text());prep={(r['B'],r['L']):r['oracle_prep_removed_us']['mean'] for r in prior['rows']}
    models={}
    for layers in (46,62):
        joint=[]
        for i in range(8):
            logs=[]
            for l in loads:
                B,L=l['B'],l['L'];s=next(r for r in p['paired'] if (r['B'],r['L'],r['window'])==(B,L,2048));f=next(r for r in p['paired'] if (r['B'],r['L'],r['window'])==(B,L,0));old=(46*s['times_ns']['A'][i]+16*f['times_ns']['A'][i])/1000
                saving=layers*prep.get((B,L),0.);logs.append(math.log(old/(old-saving)))
            joint.append(math.exp(statistics.mean(logs)))
        models[str(layers)]=dict(**stats(joint),scope=f'Scenario only: delete old RAW2 BF16-ready oracle preparation for four batched loads across {layers} layers, zero producer/ready-load cost; d1/v1 unchanged. S19 v16 actually uses FA3, and full-layer savings were not isolated. This is neither a measured fused gain nor a six-load upper bound. CI reflects baseline variation only, not oracle/model uncertainty. Replay overhead remains in layer event times.')
    out=dict(job=receipt['id'],receipt=str(path),checks=p['checks'],elapsed_s=p['elapsed_s'],base=p['identities']['base_sha256'],shapes=shape,loads=loads,free_finish_models=models,raw=p,fused_candidate=dict(status='unmeasured: global last-CTA finish cannot overlap its preceding weight stream; per-head fusion needs a new coordination/producer design',ratios={l['label']:None for l in loads}),limitations=['SASS/static throughput do not establish dynamic ALU idle.','Standalone parallel finish cost is not a rigorous lower bound on a fused epilogue.','CUPTI critical increments have profiling perturbation and are not guaranteed recoverable savings.','RAW2 altered-work preparation estimates are not actual S19 FA3 timings or a bound on unmeasured d1/v1/full fusion.'])
    (H/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    for key,r in shape.items():print(key,'attn',r['attention_critical_us'],'tail',r['finish_minus_empty_us'],'body_ns',r['finish_body_median_ns']['mean'])
    for l in loads:print(l['label'],'AA',l['layer_A_A'],'finish/empty',l['finish_probe_ratio'])
    print('FREE_FINISH_MODELS',models);return out
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();analyze(a.receipt)
