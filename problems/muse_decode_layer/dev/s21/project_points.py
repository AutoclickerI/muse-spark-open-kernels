"""Planning projection from selected per-cell stage deltas, counting replacements once."""
import json,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import paired_evidence
import compose

def main():
    chain=json.loads((R/'problems/muse_attention_decode/dev/full_attention/chain/result.json').read_text())
    dense=json.loads((R/'problems/muse_bf16_proj/dev/large_split/final_whole_results.json').read_text())
    whole={(b,q,w):statistics.mean(row['values']['A']) for row in dense['raw']['rows'] for b,q,w in [row['case']]}
    stage={}
    for label,arm in chain['preferred'].items():
        if label!='32x8f':continue  # Other FA3/VQ rows are superseded by ports.
        b,q=map(int,label[:-1].split('x'));w=2048 if label[-1]=='s' else 0;cell=chain['summary'][label]
        stage[b,q,w]=dict(arm=arm,control_us=cell['control_us'],ratio=cell['ratios'][arm]['r'],saved_us=cell['control_us']*(1-cell['ratios'][arm]['r']))
    ports=json.loads((R/'problems/muse_attention_decode/dev/public_ports/result.json').read_text())
    for (b,q,sliding),name in compose.PUBLIC_SELECTION.items():
        label=f'{b}x{q}'+('s' if sliding else 'f');arm='L' if name=='lee' else 'O'
        point=next(x for x in ports['points'] if x['cell']==label and x['arm']==arm)
        control=statistics.mean(x['A'] for x in ports['seed_ns'][label])/1000
        stage[b,q,2048 if sliding else 0]=dict(arm=arm,control_us=control,ratio=point['r'],saved_us=control*(1-point['r']))
    gate=1.0044046459819183;c3=1.000967907286277;np_v16=1.0041276519451878
    def project(baseline):
        loads={}
        for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
            old=sum(weight*baseline[b,q,w] for w,weight in ((2048,46),(0,16)))
            saved=sum(weight*stage.get((b,q,w),{}).get('saved_us',0) for w,weight in ((2048,46),(0,16)))
            ratio=old/(old-saved)
            if (b,q)==(16,8):ratio*=np_v16
            loads[f'{b}x{q}']=dict(old_load_us=old,stage_saved_us=saved,R=ratio)
        r=math.exp(statistics.mean(math.log(v['R']) for v in loads.values()))*gate*c3
        return dict(R=r,loads=loads)
    receipt=next((R/'runs/muse_decode_layer').glob('*28b5c189*.json'));official=json.loads(receipt.read_text())
    p=next((R/'runs/muse_prefill_layer').glob('*6ac9317e*.json'));s19=paired_evidence(json.loads(p.read_text())['result']['log_tail'],'S19')
    adjusted={}
    for rec in official['result']['records']:
        if rec.get('event')!='benchmark':continue
        c=rec['case'];b,q,w=c['batch'],c.get('q_len',1),2048 if c['layer']=='sliding' else 0
        label=('d' if q==1 else 'v')+str(b)+('S' if w else 'F')
        adjusted[b,q,w]=rec['median_ns']/1000/s19['shapes'][label]['r']['INC']
    out=dict(planning_only=True,stage_points=[chain['job'],ports['job']],whole_baseline=dense['job'],stage_deltas={str(k):v for k,v in stage.items()},
        primary=project(whole),official_S18_adjusted_sensitivity=project(adjusted),factors=dict(gate=gate,C3=c3,NP_verify16=np_v16),
        replacement_credit='Ports replace the prior FA_CHAIN/VQ16 savings; M64/R64 receive no separate credit. C3 beats operator d16F; v32F Rw4 remains.',
        limitations='Different runs/weights/timing boundaries; assumes additive stage savings and independent dispatch effects. G0 requires the native composed held-out R lower bound, never this projection.')
    (H/'projection.json').write_text(json.dumps(out,indent=2)+'\n');print('Point projection',out['primary']['R'],'official sensitivity',out['official_S18_adjusted_sensitivity']['R'])
if __name__=='__main__':main()
