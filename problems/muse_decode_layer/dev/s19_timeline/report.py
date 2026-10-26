"""Decode the S19 observer; all-round owner accounting, TASK roofs, gap shares."""
import argparse,ast,base64,csv,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(H.parent/'s15_timeline'),str(R/'tools')]
# Same S15 decoder, with v2 median wall retained instead of redundant median-stage vectors.
_decoder=(H.parent/'s15_timeline/decode.py').read_text()
_fn=next(n for n in ast.parse(_decoder).body if isinstance(n,ast.FunctionDef) and n.name=='decode')
_decode_source=ast.get_source_segment(_decoder,_fn).replace('wall=sum(critical.values())', "wall=seed['wall_ns']/1000").replace("for key in ('profile_wall_samples_us',):", "for key in ():")
exec(compile(_decode_source,'<s19-timeline-decoder>','exec'),globals())
from analysis import chain_account
from lab_packet import read_packets
LOADS=('d1','d16','d32','v1','v16','v32')
FIELDS=('read_bytes','write_bytes','bf16_flops','fp8_flops')
PHASES=('qkv','attention','o','pool_router','moe_route','moe_W13','moe_W2','moe_fused','moe_finalize','moe_overlap','unpool','glue','idle','busy_unassigned')
CORE=('moe_W13','moe_W2','moe_fused','moe_finalize','moe_overlap')
DISPLAY=('qkv','attention','o','pool_router','moe_route','MoE core','moe_W13','moe_W2','moe_fused','moe_finalize','moe_overlap','unpool','glue','idle','busy_unassigned')
LABELS={'qkv':'QKV + gate / finish¹','attention':'Attention¹','o':'O','pool_router':'Pool/router','moe_route':'Route (norm + expert metadata)','MoE core':'MoE core (combined envelope)','moe_W13':'  W13','moe_W2':'  W2 + inline finish','moe_fused':'  Fused W13+W2+finish','moe_finalize':'  Separate finalize','moe_overlap':'  Mixed MoE busy overlap','unpool':'Unpool','glue':'Glue / b1+b2+setup','idle':'GPU idle','busy_unassigned':'Mixed owner busy overlap'}

def binary(blob):
    raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0];values=[];v=shift=0
    for a in raw[4+size:]:
        v|=(a&127)<<shift
        if a&128:shift+=7
        else:values.append(v);v=shift=0
    assert shift==0
    return values

def add(*values):
    return {k:sum(v.get(k,0) for v in values) for k in FIELDS}

def roof(work):
    return max((work.get('read_bytes',0)+work.get('write_bytes',0))/4.8e6,
               (work.get('bf16_flops',0)+work.get('fp8_flops',0))/989e6)

def work_rows(seed,cell):
    w=seed['work'];counts,_,_=seed['fixture'];t=int(cell['case'][1:-1])*(8 if cell['case'][0]=='v' else 1)
    result={k:{} for k in PHASES}
    result['qkv']=add(*(w.get(k,{}) for k in ('qkv_gate','qkv3','gate')))
    result['attention']=add(*(v for k,v in w.items() if k.startswith('attention')))
    result['o']=w.get('o',{});result['pool_router']=w.get('pool_router',{})
    result['unpool']=add(*(w.get(k,{}) for k in ('unpooler','unpooler_slice')))
    result['moe_route']=add(w.get('route',{}),w.get('moe_meta',{}))
    result['glue']=add(*(w.get(k,{}) for k in ('b1','b2','setup')))
    result['moe_finalize']=w.get('moe_combine',{})
    n=sum(c>0 for c in counts);pairs=sum(counts)
    phases={n['phase'] for n in cell['trace']}
    core=add(*(w.get(k,{}) for k in ('moe_small','moe_light','moe_heavy','moe_combine')))
    if phases & {'moe_W13','moe_W2'}:
        result['moe_W13']=dict(read_bytes=n*(8192*4096*9//16+4+4096*2)+pairs*4096*2,write_bytes=pairs*4096*2,bf16_flops=2*pairs*4096*8192,fp8_flops=0)
        result['moe_W2']=dict(read_bytes=n*(4096*4096*9//16+4)+pairs*4096*2,write_bytes=pairs*(4096*4+64*4),bf16_flops=2*pairs*4096*4096,fp8_flops=0)
        if t<=32:
            result['moe_W2']['read_bytes']+=t*8*4*2;result['moe_W2']['write_bytes']+=t*4096*2
        assert add(result['moe_W13'],result['moe_W2'],result['moe_finalize'])==core
    else:result['moe_fused']=core
    result['MoE core']=core
    return result

def unpack(blob):
    x=decode(blob);v=binary(blob);allround=v[x['round_values_offset']:];width=x['round_width']
    assert tuple(x['phases'])==PHASES
    for c in x['cases']:
        for s in c['seeds']:
            off=s['round_offset'];s['rounds']=[]
            for r in range(x['profile_reps']):
                row=allround[off+r*width:off+(r+1)*width]
                s['rounds'].append(dict(round=r,wall_us=row[0]/1000,event_us=row[1]/1000,critical_us=dict(zip(PHASES,[z/1000 for z in row[2:]]))))
                assert abs(sum(s['rounds'][-1]['critical_us'].values())-row[0]/1000)<.008
            off+=width*x['profile_reps']
            s['plain_event_us']=[z/1000 for z in allround[off:off+4]];s['mapped_event_us']=[z/1000 for z in allround[off+4:off+8]]
    assert len(allround)==len(x['cases'])*len(x['seeds'])*(width*x['profile_reps']+8)
    return x

def summarize(x):
    shapes=[];rounds=[]
    for c in x['cases']:
        allrows=[]
        for s in c['seeds']:
            work=work_rows(s,c)
            for r in s['rounds']:
                critical=r['critical_us'];critical['MoE core']=sum(critical[k] for k in CORE)
                rows={k:dict(critical_us=critical[k],roof_us=roof(work[k]),excess_us=max(0,critical[k]-1.3*roof(work[k]))) for k in DISPLAY}
                allrows.append(dict(wall_us=r['wall_us'],rows=rows))
                for k in DISPLAY:rounds.append(dict(case=c['case'],seed=s['seed'],round=r['round'],stage=k,wall_us=r['wall_us'],event_us=r['event_us'],**rows[k]))
        rows={k:{f:statistics.mean(z['rows'][k][f] for z in allrows) for f in ('critical_us','roof_us','excess_us')} for k in DISPLAY}
        controls=dict(plain_event_us=statistics.mean(z for s in c['seeds'] for z in s['plain_event_us']),mapped_event_us=statistics.mean(z for s in c['seeds'] for z in s['mapped_event_us']),profile_event_us=statistics.mean(r['event_us'] for s in c['seeds'] for r in s['rounds']))
        controls.update(mapped_over_plain=controls['mapped_event_us']/controls['plain_event_us'],profile_over_mapped=controls['profile_event_us']/controls['mapped_event_us'])
        shapes.append(dict(case=c['case'],weight=c['weight'],wall_us=statistics.mean(z['wall_us'] for z in allrows),rows=rows,controls=controls))
    loads={}
    for load in LOADS:
        cases=[c for c in shapes if c['case'][:-1]==load];assert len(cases)==2
        loads[load]=dict(wall_us=sum(c['weight']*c['wall_us'] for c in cases)/62,rows={k:{f:sum(c['weight']*c['rows'][k][f] for c in cases)/62 for f in ('critical_us','roof_us','excess_us')} for k in DISPLAY})
    incumbent=1.848375*1.0144633683891588;gap=math.log(2.6/incumbent)
    def ceiling(savings):
        ratios=[loads[k]['wall_us']/(loads[k]['wall_us']-savings[k]) for k in LOADS]
        ln=statistics.mean(math.log(r) for r in ratios)
        return dict(R=math.exp(ln),ln_R=ln,gap_share_pct=100*ln/gap,savings_us=savings)
    def selected(rows,ls=LOADS,factor=1.3):
        if factor==1.3:return {k:sum(loads[k]['rows'][s]['excess_us'] for s in rows) if k in ls else 0 for k in LOADS}
        # Mean of positive per-round excess, so the ideal-roof table uses the same estimator.
        savings={k:0. for k in LOADS}
        for c in x['cases']:
            load=c['case'][:-1]
            if load not in ls:continue
            values=[]
            for seed in c['seeds']:
                work=work_rows(seed,c)
                for rnd in seed['rounds']:
                    v=dict(rnd['critical_us']);v['MoE core']=sum(v[k] for k in CORE)
                    values.append(sum(max(0,v[k]-factor*roof(work[k])) for k in rows))
            savings[load]+=c['weight']*statistics.mean(values)/62
        return savings
    ceilings={
        'P3a':ceiling(selected(['MoE core'],('d1','d16','d32','v1'))),
        'P3b':ceiling(selected(['MoE core'],('v16','v32'))),
        'P10_to_1.3roof':ceiling(selected(['qkv','o','pool_router','unpool'])),
        'glue_route_uncovered':ceiling(selected(['glue','moe_route'])),
    }
    # Disjoint responsibility: P5b owns full and the actual FA3 v16S call; P5 owns other sliding.
    for lane in ('P5','P5b'):
        savings={k:0. for k in LOADS}
        for c in shapes:
            assigned='P5b' if c['case'].endswith('F') or c['case']=='v16S' else 'P5'
            if assigned==lane:savings[c['case'][:-1]]+=c['weight']*c['rows']['attention']['excess_us']/62
        ceilings[lane]=ceiling(savings)
    for lane,rows,ls in [('P3a',['MoE core'],('d1','d16','d32','v1')),('P3b',['MoE core'],('v16','v32')),('P10',['qkv','o','pool_router','unpool'],LOADS),('glue_route',['glue','moe_route'],LOADS)]:
        ceilings[lane+'_ideal_roof']=ceiling(selected(rows,ls,factor=1.))
    for lane in ('P5','P5b'):
        savings={k:0. for k in LOADS}
        for c in x['cases']:
            assigned='P5b' if c['case'].endswith('F') or c['case']=='v16S' else 'P5'
            if assigned!=lane:continue
            values=[max(0,rnd['critical_us']['attention']-roof(work_rows(seed,c)['attention'])) for seed in c['seeds'] for rnd in seed['rounds']]
            savings[c['case'][:-1]]+=c['weight']*statistics.mean(values)/62
        ceilings[lane+'_ideal_roof']=ceiling(savings)
    # Explicit S19 ramp scenario, not a physical bandwidth bound. Keep first qkv ramp.
    # Five stream starts (attention, O, pool, MoE, unpool) + separate gate if present; actual idle only.
    ramps={k:0. for k in LOADS};retention={k:0. for k in LOADS};coding={k:0. for k in LOADS}
    for c,raw in zip(shapes,x['cases']):
        work=raw['seeds'][0]['work'];ramp=sum(min(2.36,c['rows'][k]['critical_us']) for k in ('attention','o','pool_router','MoE core','unpool'))
        if 'gate' in work:ramp+=min(2.36,chain_account(raw['trace'],raw['dag'])['critical_us'].get('gate',0))
        ramp+=c['rows']['idle']['critical_us'];ramps[c['case'][:-1]]+=c['weight']*ramp/62
        # Interstage boundaries: same one-store+read scenario as S15, measured S19 work/layout.
        # Each read of prior partials is charged once, plus output handoff reads/writes.
        # A separate reconstruction from logical tensor roles is retained as a scenario, not HBM counters.
        t=int(c['case'][1:-1])*(8 if c['case'][0]=='v' else 1)
        material=sum(work.get(k,{}).get('write_bytes',0) for k in ('qkv_gate','qkv3','gate','o','b1','pool_router','route','unpooler','unpooler_slice'))
        material+=work.get('attention_sliding',work.get('attention_full',{})).get('write_bytes',0)
        # moe_small's internal slot writes are not interstage materialization; replace by its final BF16 tensor.
        material+=t*4096*2
        material-=t*8*4/2  # Keep one required returned route-ID store out of the two-way saving.
        retention[c['case'][:-1]]+=c['weight']*(2*material/4.8e6)/62
    ceilings['P7_ramp_scenario']=ceiling(ramps)
    ceilings['P11_materialization_scenario']=ceiling(retention)
    ceilings['P11_materialization_plus_idle_scenario']=ceiling({k:retention[k]+loads[k]['rows']['idle']['critical_us'] for k in LOADS})
    # Recompute each positive critical-minus-roof after coding, rather than adding a
    # roof reduction that may have been hidden below a predecessor's frontier.
    coded_savings={factor:{k:0. for k in LOADS} for factor in (1.,1.3)}
    dims={'qkv_gate':(2304,8192),'qkv3':(1280,8192),'gate':(1024,8192),'o':(8192,1024),'pool_router':(4608,8192),'unpooler':(8192,4096),'unpooler_slice':(1024,4096)}
    for c in x['cases']:
        values={factor:[] for factor in coded_savings}
        for seed in c['seeds']:
            work={k:dict(v) for k,v in work_rows(seed,c).items()}
            for role,(n,k) in dims.items():
                if role not in seed['work']:continue
                wb=c['projection_storage'].get(role)
                if wb is None:
                    assert role=='gate';wb=n*k*2
                group='qkv' if role in ('qkv_gate','qkv3','gate') else 'unpool' if role.startswith('unpooler') else role
                work[group]['read_bytes']-=max(0,wb-1.5*n*k)
            for rnd in seed['rounds']:
                for factor in values:
                    values[factor].append(sum(max(0,rnd['critical_us'][k]-factor*roof(work[k])) for k in ('qkv','o','pool_router','unpool')))
        for factor in values:coded_savings[factor][c['case'][:-1]]+=c['weight']*statistics.mean(values[factor])/62
    coding={k:coded_savings[1.3][k]-ceilings['P10_to_1.3roof']['savings_us'][k] for k in LOADS}
    ceilings['P10_ideal_exact12_extra']=ceiling(coding)
    ceilings['P10_target_plus_ideal_exact12']=ceiling(coded_savings[1.3])
    ceilings['P10_ideal_roof_plus_ideal_exact12']=ceiling(coded_savings[1.])
    ceilings['P5_finish_deletion_observed']=dict(R=1.0159,ln_R=math.log(1.0159),gap_share_pct=100*math.log(1.0159)/gap)
    ceilings['P3a_wait_only_modeled']=dict(R=1.016674,ln_R=math.log(1.016674),gap_share_pct=100*math.log(1.016674)/gap)
    # Joint independent-row scenario: core envelope once, route/glue and exact12 projections.
    joint={k:sum(loads[k]['rows'][s]['excess_us'] for s in ('qkv','attention','o','pool_router','moe_route','MoE core','unpool','glue','idle','busy_unassigned')) for k in LOADS}
    ceilings['joint_to_1.3roof']=ceiling(joint)
    return dict(source_sha256=x['s19_sha256'],TASK_checks=x['TASK_checks'],seeds=x['seeds'],rounds=x['profile_reps'],incumbent_score_estimate=incumbent,target=2.6,ln_gap=gap,shapes=shapes,loads=loads,ceilings=ceilings),rounds

def markdown(report):
    lines=['# S19 completion-critical timeline','','S19 `62ea5bf4`, shipment `8b415553`. All times us per layer; load means use (46 sliding + 16 full)/62. Every one of the 3 fresh seeds × 3 rounds is included. Roof = max(logical read+write bytes / 4.8 TB/s, tensor FLOPs / 989 TFLOP/s), without a fitted intercept. Excess = mean(max(critical − 1.3×roof, 0)) over rounds. These are diagnostic resource models, not attainable speedup promises.','','¹ QKV/gate projection finish stays with its actual kernel; fused query preparation stays inside attention. W2 includes its inline norm/finalizer. All four small-M loads fuse both expert projections and finish, which CUPTI cannot separate without changing the device code. Mixed MoE overlap is busy work on multiple expert phases, never GPU idle. The combined MoE envelope owns its one unique-work roof; phase rows below it overlap that envelope and must not be added. Scalar norm/softmax/selection throughput is not covered by these byte/tensor roofs.','','## Six-load stage table','','Cells: critical / roof / excess.','','| Stage | '+' | '.join(LOADS)+' |','|---|'+'---:|'*6]
    for k in DISPLAY:
        if all(report['loads'][l]['rows'][k]['critical_us']==0 and report['loads'][l]['rows'][k]['roof_us']==0 for l in LOADS):continue
        lines.append('| '+LABELS[k]+' | '+' | '.join('/'.join(f'{report["loads"][l]["rows"][k][f]:.2f}' for f in ('critical_us','roof_us','excess_us')) for l in LOADS)+' |')
    lines+=['| Wall | '+' | '.join(f'{report["loads"][l]["wall_us"]:.2f}' for l in LOADS)+' |','','## Sliding / full shapes','','| Stage | '+' | '.join(c['case'] for c in report['shapes'])+' |','|---|'+'---:|'*12]
    for k in DISPLAY:
        lines.append('| '+LABELS[k]+' | '+' | '.join('/'.join(f'{c["rows"][k][f]:.2f}' for f in ('critical_us','roof_us','excess_us')) for c in report['shapes'])+' |')
    lines+=['','## Observer controls','','Whole-replay event times include endpoint overhead outside the first/last CUPTI activity; no scaling is applied.','','| Shape | plain | mapped | profiled | mapped/plain | profiled/mapped |','|---|---:|---:|---:|---:|---:|']
    for c in report['shapes']:
        v=c['controls'];lines.append('| '+c['case']+' | '+' | '.join(f'{v[k]:.5f}' for k in ('plain_event_us','mapped_event_us','profile_event_us','mapped_over_plain','profile_over_mapped'))+' |')
    lines+=['','## Counterfactual gap shares','',f'Lab-transported S19 estimate {report["incumbent_score_estimate"]:.6f}x = 1.848375 × 1.014463368; scored S19 is pending. Close target 2.6x; ln gap {report["ln_gap"]:.6f}. One lever at a time; shares do not add.','','| Scenario | R ceiling / observation | % ln gap |','|---|---:|---:|']
    for k,v in report['ceilings'].items():lines.append(f'| {k} | {v["R"]:.6f} | {v["gap_share_pct"]:.2f}% |')
    return '\n'.join(lines)+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--local',action='store_true');a=p.parse_args()
    if a.local:
        packet=json.loads(a.receipt.read_text())['packet'];record=None
    else:
        rec=json.loads(a.receipt.read_text());records=read_packets(rec['result']['log_tail']);record=next(r for r in records if r['packet'].get('kind')=='S19_TIMELINE');packet=record['packet']
    source_audit={}
    if not a.local:
        sent=a.receipt.with_suffix('.py').read_text();tree=ast.parse(sent)
        payloads=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets))
        name,source_sha,payload=next(p for p in payloads if p[0]==record['name'])
        manifest=json.loads((H/'manifest.json').read_text())
        digest=hashlib.sha256(payload.encode()).hexdigest()
        assert digest==manifest['payload_sha256'] and source_sha==record['source_sha256']==digest[:16]
        assignment=next(n for n in ast.parse(payload).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_body' for t in n.targets))
        encoded=next(n.value for n in ast.walk(assignment.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000)
        body=lzma.decompress(base64.b85decode(encoded)).decode()
        literals={t.id:ast.literal_eval(n.value) for n in ast.parse(body).body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) for t in n.targets if isinstance(t,ast.Name)}
        for key,field in [('S19_SOURCE','source_sha256'),('S19_CHECKER','checker_sha256'),('T15_ACTIVITY','observer_sha256')]:
            assert hashlib.sha256(literals[key].encode()).hexdigest()==manifest[field]
        source_audit=dict(submitted_wrapper_sha256=hashlib.sha256(sent.encode()).hexdigest(),payload_sha256=digest,source_checker_observer_pins_match=True)
    assert packet['complete'] and not packet.get('transport_errors')
    blob=next(l.split(' ',1)[1] for l in packet['lines'] if l.startswith('S19_TIMELINE_XZ85 '));x=unpack(blob)
    report,rounds=summarize(x);a.out.mkdir(exist_ok=True,parents=True)
    (a.out/'timeline.json.xz').write_bytes(lzma.compress(json.dumps(x,separators=(',',':')).encode(),preset=9))
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.out/'H200.md').write_text(markdown(report))
    with (a.out/'rounds.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rounds[0]));w.writeheader();w.writerows(rounds)
    (a.out/'rounds.csv.xz').write_bytes(lzma.compress((a.out/'rounds.csv').read_bytes(),preset=9))
    validation=dict(source_sha256=x['s19_sha256'],TASK_checks=x['TASK_checks'],schedule_identical=all(c['schedule_identical'] for c in x['cases']),all_stamps_monotone=x['all_stamps_monotone'],max_accounting_error=x['max_accounting_error'],shapes=len(x['cases']),seeds=len(x['seeds']),rounds=x['profile_reps'],packet_checksum=record['checksum'] if record else None,local_sm120_only=a.local,receipt=str(a.receipt.relative_to(R) if a.receipt.is_absolute() else a.receipt),source_audit=source_audit)
    (a.out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation));print(json.dumps(report['ceilings'],indent=2))
if __name__=='__main__':main()
