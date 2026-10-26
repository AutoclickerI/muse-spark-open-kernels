"""S15 trace-derived continuity bounds, without pretending CUPTI activity is an HBM counter."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3]
T=R/'problems/muse_decode_layer/dev/s15_timeline'
sys.path.insert(0,str(T))
from analysis import chain_account
from decode import decode,gap_audit

BPS=4.58e12
RAMP=2.36
OURS=1.8378866426354703
LEADER=2.3601
LOADS=('d1','d16','d32','v1','v16','v32')
GROUPS={
 'front':('qkv_gate','qkv3'),
 'qkv -> attention':('attention_sliding','attention_full'),
 'attention -> gate':('gate',),
 'attention/gate -> O':('o',),
 'O -> b1':('b1',),
 'b1 -> pool/router':('pool_router',),
 'pool/router -> route':('route',),
 'route -> MoE':('moe_small','moe_light','moe_heavy','moe_meta','moe_combine'),
 'MoE -> unpooler':('unpooler','unpooler_slice'),
 'unpooler -> b2':('b2',),
}
MAJOR={s for k,v in GROUPS.items() if k not in ('O -> b1','pool/router -> route','unpooler -> b2') for s in v}

def overlap(a,b,c,d):return max(0,min(b,d)-max(a,c))

def analyze(x):
    shapes=[];gaplog=math.log(LEADER/OURS)
    for c in x['cases']:
        a=chain_account(c['trace'],c['dag']);audit=gap_audit(c)
        assert a['accounting_error_fraction']<1e-10
        by={n['node']:n for n in c['trace']}
        seed=next(s for s in c['seeds'] if s['seed']==c['trace_seed']);rows={};intervals=[]
        for p in a['path']:
            n=by[p['node']];end=n['end_ns']/1000;start=end-p['critical_us']
            intervals.append(dict(stage=p['stage'],critical_start_us=start,end_us=end,
                                  kernel_start_us=n['start_ns']/1000,kernel_end_us=end,
                                  pre_frontier_overlap_us=max(0,start-n['start_ns']/1000),
                                  gap_us=p['gap_us']))
        for boundary,roles in GROUPS.items():
            nodes=[n for n in c['trace'] if n['role'] in roles]
            if not nodes:continue
            cost=sum(a['critical_us'].get(s,0) for s in roles)
            # Attribute busy logical gaps to the sole boundary owner present.
            cost+=sum(v for k,v in audit['busy_overlap_us'].items() if set(k.split('+'))<=set(roles))
            read=sum(seed['work'].get(s,{}).get('read_bytes',0) for s in roles)
            begin=min(n['start_ns'] for n in nodes)/1000
            early=sum(overlap(p['critical_start_us'],p['end_us'],begin,begin+RAMP) for p in intervals if p['stage'] in roles)
            # One fitted intercept per logical stage. This is an explicit
            # model ceiling, not a measured duration of griddepcontrol.wait.
            cap=min(RAMP,cost) if boundary not in ('front','O -> b1','pool/router -> route','unpooler -> b2') else 0.
            service=read/BPS*1e6
            boundary_idle=0.
            for step in a['path']:
                if step['stage'] not in roles or not step['gap_us']:continue
                right=by[step['node']]['start_ns'];left=right-round(step['gap_us']*1000)
                cuts=sorted({left,right}|{max(left,min(right,n[k])) for n in c['trace'] for k in ('start_ns','end_ns')})
                boundary_idle+=sum((hi-lo)/1000 for lo,hi in zip(cuts,cuts[1:]) if not any(n['start_ns']<hi and n['end_ns']>lo for n in c['trace']))
            rows[boundary]=dict(critical_us=cost,actual_gpu_idle_us=boundary_idle,model_read_bytes=read,
                full_rate_equivalent_us=service,unique_model_TBps=read/max(cost,1e-12)/1e6,
                kernel_start_ramp_exposed_us=min(cap,early),
                deferred_ramp_ceiling_us=cap,
                service_deficit_us=max(0,cost-service),
                first_kernel_us=begin,last_kernel_us=max(n['end_ns'] for n in nodes)/1000)
        # Exact activity-derived interval union; no rate is inferred here.
        cuts=sorted({n[k] for n in c['trace'] for k in ('start_ns','end_ns')})
        no_major=0.;no_major_intervals=[]
        for left,right in zip(cuts,cuts[1:]):
            active=[n for n in c['trace'] if n['start_ns']<right and n['end_ns']>left]
            if not any(n['role'] in MAJOR for n in active):
                no_major+=(right-left)/1000
                no_major_intervals.append([left/1000,right/1000,sorted({n['role'] for n in active})])
        # P11: a broad one-write/one-read elimination scenario. It includes
        # projection partials plus b1 h/s1, route x/weights/IDs (IDs must still
        # be returned), and final MoE output. Initial inputs, returned b2 outputs
        # and externally visible KV updates are excluded. Every byte is assumed
        # cold to HBM; L2 hits and overlap reduce the actual opportunity.
        edge_stages=('qkv_gate','qkv3','gate','attention_sliding','attention_full','o','b1','pool_router','route','unpooler','unpooler_slice')
        edge_bytes={k:2*seed['work'].get(k,{}).get('write_bytes',0) for k in edge_stages}
        tokens=int(c['case'][1:-1])*(8 if c['case'][0]=='v' else 1)
        edge_bytes['MoE output']=2*tokens*4096*2
        # Keep one required store of returned route IDs.
        edge_bytes['route']=max(0,edge_bytes['route']-tokens*8*4)
        retained_bytes=sum(edge_bytes.values())
        retained_us=retained_bytes/BPS*1e6
        ramp=sum(v['deferred_ramp_ceiling_us'] for v in rows.values())
        early=sum(v['kernel_start_ramp_exposed_us'] for v in rows.values())
        idle=audit['gpu_idle_us']
        shapes.append(dict(case=c['case'],weight=c['weight'],trace_seed=c['trace_seed'],
            wall_us=a['wall_us'],gpu_idle_us=idle,no_major_kernel_resident_us=no_major,
            retained_edge_bytes=edge_bytes,retained_operand_bytes=retained_bytes,retained_operand_HBM_equivalent_us=retained_us,
            no_major_intervals=no_major_intervals,rows=rows,critical_intervals=intervals,
            kernel_start_ramp_exposed_us=early,deferred_ramp_ceiling_us=ramp,
            ramp_plus_idle_ceiling_us=min(a['wall_us'],ramp+idle)))
    loads={}
    for load in LOADS:
        cs=[c for c in shapes if c['case'][:-1]==load];assert len(cs)==2
        keys=('wall_us','gpu_idle_us','no_major_kernel_resident_us','kernel_start_ramp_exposed_us','deferred_ramp_ceiling_us','ramp_plus_idle_ceiling_us','retained_operand_bytes','retained_operand_HBM_equivalent_us')
        v={k:sum(c['weight']*c[k] for c in cs)/62 for k in keys}
        for k in keys[1:]:v[k+'_wall_pct']=100*v[k]/v['wall_us']
        v['load_total_wall_us']=62*v['wall_us']
        v['load_total_continuity_ceiling_us']=62*v['ramp_plus_idle_ceiling_us']
        v['gap_share_pct']=100*math.log(v['wall_us']/(v['wall_us']-v['ramp_plus_idle_ceiling_us']))/(6*gaplog)
        v['boundary_gpu_idle_us']={b:sum(c['weight']*c['rows'].get(b,{}).get('actual_gpu_idle_us',0) for c in cs)/62 for b in GROUPS}
        v['boundary_reads_MB']={b:sum(c['weight']*c['rows'].get(b,{}).get('model_read_bytes',0) for c in cs)/62/1e6 for b in GROUPS}
        v['boundary_critical_us']={b:sum(c['weight']*c['rows'].get(b,{}).get('critical_us',0) for c in cs)/62 for b in GROUPS}
        v['boundary_us']={b:sum(c['weight']*c['rows'].get(b,{}).get('deferred_ramp_ceiling_us',0) for c in cs)/62 for b in GROUPS if b!='front'}
        v['service_deficit_us']={b:sum(c['weight']*c['rows'].get(b,{}).get('service_deficit_us',0) for c in cs)/62 for b in GROUPS}
        loads[load]=v
    ceiling_R=math.exp(sum(math.log(v['wall_us']/(v['wall_us']-v['ramp_plus_idle_ceiling_us'])) for v in loads.values())/6)
    return dict(receipt='7eaa262a30ec483e',s15_sha256=x['s15_sha256'],
        basis=dict(timestamp='One retained median-seed trace per shape; all streams; 46/16 weighted; completion-chain increments conserve wall.',
            gpu_idle='Exact union with no recorded kernel/memcpy/memset. Not an HBM counter.',
            no_major='No major weight/KV-stage kernel resident; glue/vector traffic can remain. Not an HBM utilization measurement.',
            ramp='Model ceiling: at most one 2.36us fitted ramp per exposed weight/KV stage; glue ramps excluded, clipped to its completion-critical time; first qkv ramp retained, concurrent MoE branches combined. Add only actual GPU idle, not busy DAG gaps.',
            early='Scenario only: ramp begins at kernel start, then intersect with completion-critical intervals. Actual load issue and PDL wait placement are unobserved by CUPTI activity.',
            deficit='Unique-read service-equivalent headroom at4.58TB/s; padding/cache/activation reuse and read multiplicity make this an optimistic model, not measured HBM idle.',
            gap='Per-load contribution: ln(W/(W-bound))/(6 ln(leader/ours)); sums across the six equally weighted log loads.',
            retained_operands='One store+read per modeled interstage output, excluding b2 returns and one route-ID return store; add final MoE output. HBM service-equivalent scenario, not measured miss traffic. Projection partials included; internal MoE repeated activation reads excluded. Not a strict physical bound on retained-operand latency.',
            hardware_HBM_idle='Not identifiable from activity timestamps/unique-byte models; no per-kernel DRAM throughput or instruction timestamps are present.'),
        stream_TBps=BPS/1e12,intercept_us=RAMP,current_score=OURS,leader_score=LEADER,
        retained_HBM_equivalent_R=math.exp(sum(math.log(v['wall_us']/(v['wall_us']-v['retained_operand_HBM_equivalent_us'])) for v in loads.values())/6),
        model_ceiling_R=ceiling_R,model_ceiling_gap_share_pct=100*math.log(ceiling_R)/gaplog,
        shapes=shapes,loads=loads)

def table(x):
    lines=['# S15 cross-stage continuity bound','',
      'CUPTI activity has no HBM counter. These are exact activity intervals plus explicitly modeled ramp ceilings, not measured HBM idle. Times are weighted us/layer: (46 sliding +16 full)/62 at each load; multiply by62 for the full scored load.',
      '', '| Load | Wall us | GPU idle us | No major kernel resident us | Early-ramp scenario us | Weight/KV ramp+idle ceiling us (% wall) | Contribution to snapshot GOAL gap |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in x['loads'].items():lines.append(f"|{k}|{v['wall_us']:.2f}|{v['gpu_idle_us']:.2f}|{v['no_major_kernel_resident_us']:.2f}|{v['kernel_start_ramp_exposed_us']:.2f}|{v['ramp_plus_idle_ceiling_us']:.2f} ({v['ramp_plus_idle_ceiling_us_wall_pct']:.1f}%)|{v['gap_share_pct']:.1f}%|")
    lines+=['',f"Joint model ceiling R={x['model_ceiling_R']:.5f}, {x['model_ceiling_gap_share_pct']:.1f}% of ln(2.3601/1.83789). This is not a demonstrated speedup or a physical HBM utilization bound.",'',
       '| Boundary | '+'|'.join(LOADS)+'|','|---|'+'---:|'*6]
    for b in GROUPS:
        if b=='front':continue
        lines.append('|'+b+'|'+'|'.join(f"{x['loads'][l]['boundary_us'].get(b,0):.2f}" for l in LOADS)+'|')
    lines+=['', 'Logical read MB / completion-critical us, with busy same-owner DAG gaps assigned to that boundary. These are unique-byte work models, not DRAM byte counters or additive isolated-stage timings.', '', '| Boundary | '+'|'.join(LOADS)+'|','|---|'+'---:|'*6]
    for b in GROUPS:lines.append('|'+b+'|'+'|'.join(f"{x['loads'][l]['boundary_reads_MB'][b]:.2f} / {x['loads'][l]['boundary_critical_us'][b]:.2f}" for l in LOADS)+'|')
    lines+=['', 'Measured no-GPU-activity gaps by the next boundary (us); these exclude all gaps covered by another stream.', '', '| Boundary | '+'|'.join(LOADS)+'|','|---|'+'---:|'*6]
    for b in GROUPS:
        if any(x['loads'][l]['boundary_gpu_idle_us'][b] for l in LOADS):lines.append('|'+b+'|'+'|'.join(f"{x['loads'][l]['boundary_gpu_idle_us'][b]:.3f}" for l in LOADS)+'|')
    lines+=['', '| Load | Interstage materialization MB | HBM service equivalent us | % wall |', '|---|---:|---:|---:|']
    for l,v in x['loads'].items():lines.append(f"|{l}|{v['retained_operand_bytes']/1e6:.3f}|{v['retained_operand_HBM_equivalent_us']:.3f}|{v['retained_operand_HBM_equivalent_us_wall_pct']:.2f}%|")
    lines+=['','P11 retention scenario counts one store/read per interstage output at 4.58TB/s. Already-L2 hits reduce HBM savings; on-chip retention might additionally save L2 issue/latency or change inner-loop multiplicity. Those are not bounded by this trace and need the P11 oracle. The known MoE 171/403MB repeated activation-read estimates at v16/v32 are internal-stage multiplicity, not an additive cross-stage cold-HBM stream saving.','', f"Pure materialization HBM-equivalent joint R={x['retained_HBM_equivalent_R']:.5f}; it cannot be added to the ramp ceiling because removed latency/traffic can overlap.",'', 'The early scenario assumes the ramp starts at kernel entry; the deferred ceiling permits independent loads to have been sunk until their dependency frontier. The spread is precisely what SASS/load-order audits and the paired arms test. Internal MoE deficits cannot be assigned to cross-stage continuity from unique bytes alone.','',
      'The no-major-kernel and GPU-idle rows are different quantities. Resident kernels can wait or compute; other streams can stream while the logical critical chain waits. No busy MoE gap is relabeled as an idle launch gap.']
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,default=R/'runs/muse_decode_layer/20261026T001048Z_check_7eaa262a30ec483e.json');a=p.parse_args()
    text=json.loads(a.receipt.read_text())['result']['log_tail']
    blob=next(l.split('S15_TIMELINE_XZ85 ',1)[1] for l in text.splitlines() if 'S15_TIMELINE_XZ85 ' in l)
    x=analyze(decode(blob));x['receipt_sha256']=hashlib.sha256(a.receipt.read_bytes()).hexdigest()
    (H/'bound.json').write_text(json.dumps(x,indent=2)+'\n');(H/'BOUND.md').write_text(table(x));print(table(x))
