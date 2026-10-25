"""Offline, source-pinned 12-shape lever model; refresh from landed S6/S5/S4/S3/S2/S1 profiles."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import argparse, hashlib, json, math, re, sys
R=Path(__file__).resolve().parents[4];H=Path(__file__).parent;P=R/'problems/muse_decode_layer/profiles'
sys.path.insert(0,str(R))
from sim import bound, hw
RESET='c0e74bedf7b74be5';SCALE='58ae2e2cf8df45ef';W2='97be11d4631b44d6'
# MoE kernels by family: v9 NT1 (moe_fused), small-M flag variants (moe_var, S9), Humming/portfolio, S5-S7 H_H (idist,
# pmnt), S6 mixed_parallel (mwfx/mwrc/mwwi, fx_), S7R R_full and S9 N_full branches (mwbr gemm/route_live/finish; N_full's
# NT1 side is pmz::phase).
MOE_KERNELS=('moe_fused','moe_var<','humming<','_pf_','_best6_pool','_best6_positions','mwfx::','mwrc::','mwwi::','mwbr::','idist::','pmnt::','pmz::','pm_meta_kernel','fx_')
# MoE engines that fork side-stream branches (S6 mixed_parallel W13; S7R R_full and S9 N_full W13 and W2): the profile
# sums both branches' durations, so the overlap (total minus 'GPU busy') comes off the MoE stage.
FORKED_MOE=('mixed_parallel_m256',)
STAGES=('qkv_gate','o','pool_router','unpooler','attention','moe','boundaries','route','other')
ROLES=STAGES[:4]
NAMES=dict(qkv_gate='QKV+gate',o='O',pool_router='Pool+router',unpooler='Unpool',attention='Attention',moe='MoE',boundaries='Boundaries x2',route='Route+norm',other='Other/unreported')
CANDIDATES=(('S9', 'cand_r3_best7_s9', 'f55f616254a88e33781e897de0a8bfbfb82a410d1a03f1da8e8cae0f2de23f6e'), ('S8', 'cand_r3_best7_s8', 'cb84645d1af8b813529a64cfe746dd5846160860deebbe8afd8154007a23c366'), ('S7R', 'cand_r3_best7_s7r', '56c51fa026210357f9865d114e6bf7e609c52049e40b263f589d23ade95a8101'), ('S7', 'cand_r3_best7_s7', 'bd3932d1cc3c0c82d60145772148fd19fc779465dd360bda610d214385401390'), ('S6', 'cand_r3_best7_s6', '4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f'), ('S5', 'cand_r3_best7_s5', 'f7064fc0127e80a955c7dc9aa5c68eb6aefada179c2841c5c58bce8dabd20284'), ('S4', 'cand_r3_best7_s4', '9b2b2a48337aa13ba1daa45eec8a30b566cc3e59c7e1bb317d39695413d3404e'), ('S3', 'cand_r3_best7_scale_recoded_m256', 'cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7'), ('S2', 'cand_r3_best7_scale_w2fix', 'b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337'), ('S1', 'cand_r3_best7_scale', '42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7'))


def read_receipt(job):
    paths=sorted((R/'runs').glob('*/*_'+job+'.json'))
    if len(paths)!=1:raise ValueError(f'exactly one saved receipt required for {job}: {paths}')
    p=paths[0];d=json.loads(p.read_text());return p,d['result']


def deltas():
    sp,s=read_receipt(SCALE);wp,w=read_receipt(W2)
    scale={int(m):float(v) for m,v in re.findall(r'MEAN M(\d+) .*?save_us=([\d.]+)',s['log_tail'])}
    assert set(scale)=={8,16,32,128}
    times={n:float(v) for n,v in re.findall(r'AGG (\w+) n=12 mean=([\d.]+)',w['log_tail'])}
    assert {'masked','CTA1g2112'}<=times.keys()
    return scale,times['masked']-times['CTA1g2112'],[(sp,s),(wp,w)]


def profile_file(path):
    d=json.loads(path.read_text())
    assert d['summary']['passed'] and len(d['profiles'])==len(d['benchmarks'])==12
    assert all(b.get('status')=='pass' for b in d['benchmarks'])
    return d


def source_tag(d):
    if d['job_id']==RESET:return 'reset'
    matches=sorted((R/'runs/muse_decode_layer').glob('*_'+d['job_id']+'.json'))
    if not matches:raise ValueError('Accepted source receipt is required to identify the bank contract')
    receipt=json.loads(matches[-1].read_text())
    # okc_run saves the official response here; the local SHA is in the ledger,
    # not normally in that response. Hash the immutable submitted source itself.
    submitted_sha=hashlib.sha256(matches[-1].with_suffix('.py').read_bytes()).hexdigest()
    if 'sha256' in receipt:assert receipt['sha256']==submitted_sha
    for tag,stem,sha in CANDIDATES:
        if submitted_sha==sha:return tag
    raise ValueError('Unsupported source SHA: identify which deltas are already integrated before using it')


def choose_source(explicit):
    if explicit:
        p=explicit.resolve();d=profile_file(p)
        return source_tag(d),p,d
    for tag,stem,sha in CANDIDATES:
        for p in sorted(P.glob(stem+'.*.json'),reverse=True):
            d=profile_file(p)
            assert source_tag(d)==tag,(p,tag)
            return tag,p,d
    p=next(P.glob('*.'+RESET+'.json'));return 'reset',p,profile_file(p)


def correction_for(tag,tokens,scale,w2):
    if tag!='reset' and tag not in {c[0] for c in CANDIDATES}:raise ValueError(tag)
    return (scale.get(tokens,0.) if tag=='reset' else 0.,
            w2 if tokens==256 and tag in ('reset','S1') else 0.)


def plan_for(d,path):
    # Only actual final runtime choices; inherited tuner summaries may disagree.
    plans={}
    for text in path.with_suffix('.log').read_text().splitlines():
        if text.startswith('BEST7_PLAN '):
            row=json.loads(text.removeprefix('BEST7_PLAN '));plans[tuple(row['key'])]=row
    assert len(plans)==12
    return plans


def floor_data(case,bench):
    t=case['batch']*case.get('q_len',1)
    works={n:bound.projection_work(dict(m=t,layer=n),activation_factor=0) for n in ('qkv_gate','o','pooler','router','unpooler')}
    works['pool_router']=works['pooler']+works['router']
    # Cold, single-read logical weights. BF16 hi+lo router retains its four bytes.
    floors={n:max(works[n].nbytes/hw.HBM_BPS,works[n].bf16_flops/hw.BF16_FLOPS)*1e6 for n in ROLES}
    touched,routes=bound.inferred_routing('muse_decode_layer',case,bench['speed_of_light_ns']/1e3)
    a=bound.attention_work(case['batch'],case.get('q_len',1),case['ctx'],case['layer'],external_io=False)
    m=bound.nvfp4_expert_work(touched,routes,t,external_io=False)
    resource=lambda w:max(w.nbytes/hw.HBM_BPS,(w.bf16_flops/hw.BF16_FLOPS+w.fp8_flops/hw.FP8_FLOPS))*1e6
    floors.update(attention=resource(a),moe=resource(m),boundaries=(5*8192*2)/hw.HBM_BPS*1e6,
                  route=(4096*2+256*4)/hw.HBM_BPS*1e6,other=None)
    return floors,dict(touched=touched,routes=routes,context=case['ctx'],projection_bytes={n:works[n].nbytes for n in ROLES},
                       attention_bytes=a.nbytes,moe_bytes=m.nbytes)


def parse_profile(report,plan,tokens,floors):
    values={k:0. for k in STAGES};groups=[];kernels=[]
    pattern=r'^(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)\s*$'
    for line in report.splitlines():
        match=re.match(pattern,line)
        if not match:continue
        name,us,calls=match[1],float(match[2]),int(match[3]);kernels.append(dict(name=name,us=us,calls=calls))
    assert kernels
    custom=[];fixed=[];vdsk=[]
    for role,choice in plan['projection'].items():
        normalized='unpooler' if role=='unpooler_slice' else role
        if choice=='custom' and tokens<=64 and normalized!='unpooler':custom.append(normalized)
        elif choice=='custom' and tokens<=8 and normalized=='unpooler':custom.append(normalized)
        elif isinstance(choice,(tuple,list)) and choice and choice[0]=='vdsk':vdsk.append(normalized)
        elif isinstance(choice,(tuple,list)) and tuple(choice)==('stream',6,1):
            assert (tokens,normalized)==(1,'o'),(tokens,normalized,choice)
            fixed.append(normalized)
    groups_by={}
    for k in kernels:
        n,u=k['name'],k['us']
        if '_gemv_kernel' in n:
            assert custom,(plan,n)
            groups_by.setdefault(tuple(custom),dict(us=0.,names=[]))
            g=groups_by[tuple(custom)];g['us']+=u;g['names'].append(n)
        elif '_vdsk_gemm' in n:
            assert vdsk,(plan,n)
            g=groups_by.setdefault(tuple(vdsk),dict(us=0.,names=[]));g['us']+=u;g['names'].append(n)
        elif 'gf::direct<' in n:
            assert fixed==['o'],(plan,n)
            g=groups_by.setdefault(tuple(fixed),dict(us=0.,names=[]));g['us']+=u;g['names'].append(n)
        elif any(x in n for x in ('nvjet_','cublasLt::splitKreduce')):
            roles=tuple(r for r in ROLES if r not in custom and r not in fixed and r not in vdsk)
            assert roles,(plan,n)
            g=groups_by.setdefault(roles,dict(us=0.,names=[]));g['us']+=u;g['names'].append(n)
        elif any(x in n for x in MOE_KERNELS):values['moe']+=u
        elif any(x in n for x in ('attn_main','attn_combine','flash::','_qk_prep_kernel','_norm_gate_kernel')):values['attention']+=u
        elif any(x in n for x in ('_boundary_','_best7_old_boundary','boundary_cl<')):values['boundaries']+=u
        elif any(x in n for x in ('_route_','_best7_old_route','route_cl<')):values['route']+=u
        else:values['other']+=u
    assert sum(k['calls'] for k in kernels if '_gemv_kernel' in k['name'])==len(custom),(tokens,custom,kernels)
    assert sum(k['calls'] for k in kernels if 'gf::direct<' in k['name'])==len(fixed),(tokens,fixed,kernels)
    assert sum(k['calls'] for k in kernels if '_vdsk_gemm' in k['name'])==len(vdsk),(tokens,vdsk,kernels)
    assert sum(k['calls'] for k in kernels if 'nvjet_' in k['name'])==4-len(custom)-len(fixed)-len(vdsk),(tokens,custom,fixed,vdsk,kernels)
    role_bounds={r:[0.,0.] for r in ROLES};estimated=[]
    for roles,g in groups_by.items():
        denom=sum(floors[r] for r in roles)
        for role in roles:
            values[role]+=g['us']*floors[role]/denom
            role_bounds[role]=[g['us'],g['us']] if len(roles)==1 else [0.,g['us']]
            if len(roles)>1:estimated.append(role)
        groups.append(dict(roles=roles,**g,allocation='exact role' if len(roles)==1 else 'modeled proportional resource floor'))
    total=float(re.search(r'^total\s+([\d.]+) µs',report,re.M)[1]);shown=sum(k['us'] for k in kernels)
    truncated=re.search(r'^\.\.\. (\d+) more',report,re.M)
    missing=max(0.,total-shown) if truncated else 0.
    values['other']+=missing
    busy=re.search(r'^GPU busy.*?([\d.]+) µs',report,re.M)
    overlap=total-float(busy[1]) if busy and plan['moe'] in FORKED_MOE else 0.
    values['moe']-=overlap;total-=overlap
    assert abs(sum(values.values())-total)<=.05*(len(kernels)+1)+1e-9,(values,total)
    return values,groups,role_bounds,estimated,missing,total,kernels,overlap


def overall_gain(rows,loads,saved):
    totals=defaultdict(float)
    for row in rows:totals[row['case']['load']]+=row['case']['weight']*saved(row)
    assert all(loads[k]-v>0 for k,v in totals.items()),totals
    return 100.*math.expm1(sum(math.log(loads[k]/(loads[k]-v)) for k,v in totals.items())/6)


def stage_saving(row,stage,kind='estimate'):
    if stage=='other':return 0.
    time=row['stages_us'][stage]
    if kind in ('min','max') and stage in ROLES:time=row['role_bounds_us'][stage][kind=='max']
    if kind=='max' and stage in ('moe','attention'):time+=row['unreported_us']
    return max(0.,time-1.3*row['floors_us'][stage])


def make_model(tag,path,d):
    scale,w2,lab_sources=deltas();plans=plan_for(d,path);rows=[];loads=defaultdict(float);base_loads=defaultdict(float);baseline=defaultdict(float)
    for p,b in zip(d['profiles'],d['benchmarks']):
        assert p['case']==b['case'];c=dict(p['case']);t=c['batch']*c.get('q_len',1)
        key=c['batch'],c.get('q_len',1),2048 if c['layer']=='sliding' else 0
        f,work=floor_data(c,b)
        v,groups,ranges,estimated,missing,total,kernels,overlap=parse_profile(p['report'],plans[key],t,f)
        delta_scale,delta_w2=correction_for(tag,t,scale,w2)
        delta=delta_scale+delta_w2
        v['moe']-=delta
        assert v['moe']>0
        row=dict(index=p['index'],case=c,tokens=t,stages_us=v,floors_us=f,work=work,role_bounds_us=ranges,
                 estimated_roles=estimated,projection_groups=groups,unreported_us=missing,moe_branch_overlap_us=overlap,profile_total_us=total,
                 reset_profile_total_after_deltas_us=total-delta,base_adjusted_us=b['median_ns']/1000.,modeled_adjusted_us=b['median_ns']/1000.-delta,
                 delta_scale_us=delta_scale,delta_m256_us=delta_w2,base_moe_us=v['moe']+delta,benchmark_baseline_us=b['baseline_ns']/1000.,
                 kernels=kernels,plan=plans[key])
        row['profile_vs_adjusted_residual_us']=row['modeled_adjusted_us']-(total-delta)
        rows.append(row);loads[c['load']]+=c['weight']*row['modeled_adjusted_us'];base_loads[c['load']]+=c['weight']*row['base_adjusted_us'];baseline[c['load']]+=c['weight']*b['baseline_ns']/1000.
    assert len(rows)==12 and len(loads)==6
    source_score=math.prod(baseline[k]/base_loads[k] for k in loads)**(1/6)
    assert math.isclose(source_score,d['summary']['speedup'],rel_tol=1e-10),(source_score,d['summary']['speedup'])
    stage_gains={s:dict(estimate=overall_gain(rows,loads,lambda r,s=s:stage_saving(r,s)),
                        minimum=overall_gain(rows,loads,lambda r,s=s:stage_saving(r,s,'min')),
                        maximum=overall_gain(rows,loads,lambda r,s=s:stage_saving(r,s,'max')))
                 for s in STAGES if s!='other'}
    for row in rows:
        row['stage_gain_pct']={s:overall_gain(rows,loads,lambda r,s=s:stage_saving(r,s) if r is row else 0.) for s in STAGES if s!='other'}
    levers=[('Dense projections, all four roles','proj GEMV',lambda r:True,ROLES),
            ('Attention, all scored shapes','attention',lambda r:True,('attention',)),
            ('Small-M MoE M1/8/16/32','small-M MoE',lambda r:r['tokens']<=32,('moe',)),
            ('MoE M128','MoE M128',lambda r:r['tokens']==128,('moe',)),
            ('MoE M256','free M256',lambda r:r['tokens']==256,('moe',)),
            ('Boundaries and route','glue',lambda r:True,('boundaries','route'))]
    ranking=[]
    for name,owner,predicate,stages in levers:
        save=lambda r:sum(stage_saving(r,s) for s in stages) if predicate(r) else 0.
        ranking.append(dict(lever=name,owner=owner,gain_pct=overall_gain(rows,loads,save),
                            gain_max_unreported_pct=overall_gain(rows,loads,lambda r:save(r)+(r['unreported_us'] if predicate(r) and any(s in ('moe','attention') for s in stages) else 0.)),
                            saved_weighted_us=sum(r['case']['weight']*save(r) for r in rows)))
    ranking.sort(key=lambda x:x['gain_pct'],reverse=True)
    return dict(mode=tag,profile_source=str(path.relative_to(R)),profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),job=d['job_id'],
                baseline_source_speedup=source_score,modeled_speedup=math.prod(baseline[k]/loads[k] for k in loads)**(1/6),
                loads_us=loads,unadjusted_loads_us=base_loads,stage_gains=stage_gains,ranking=ranking,rows=rows,
                corrections=dict(scale_job=SCALE,scale_saved_us=scale,m256_job=W2,m256_saved_us=w2,m1='no matched M1 delta; no invented gain'),
                model_sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),R/'sim/hw.py',R/'sim/bound.py',R/'sim/muse.py',R/'sim/problems.py')},
                lab_receipts=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,_ in lab_sources])


def shape_label(r):return r['case']['load']+' '+('S' if r['case']['layer']=='sliding' else 'F')


def render(m):
    lines=['# Decode P4 — current stage levers','',
           '## Ranking: one lane at 1.3x its floor','',
           f'**Basis:** H200 profile `{m["job"]}`; mode **{m["mode"]}**. '+
           ('Until S1–S6 profiles land, this is **best7_reset plus measured scale-bank and raw M256 deltas**, not a measured candidate profile.' if m['mode']=='reset' else f'Current **{m["mode"]}** profile; already-integrated deltas are not applied again.'),'',
           '| Rank | Lever | Owning lane | Overall score gain |','|---|---|---|---:|']
    for i,row in enumerate(m['ranking'],1):lines.append(f'| {i} | {row["lever"]} | {row["owner"]} | +{row["gain_pct"]:.2f}% |')
    lines+=['',
        'Ranks 1-5 are the current five levers. Each row changes only that lane\'s shapes/stages; '
        'gains are independent counterfactuals and must not be added. Projection totals and the lane ranking '
        'do not depend on the modeled per-role split. MoE/attention gains use visible rows only; '
        'unreported rows can raise them (bounds below). These are optimistic profile-duration headrooms, '
        f'not measured speedups. Ranks3-6 span {m["ranking"][2]["gain_pct"]-m["ranking"][5]["gain_pct"]:.2f} percentage points: treat them as close '
        'priorities under router/profile/overlap uncertainty, not a resolved performance ordering.','',
        '## What is measured and what is modeled','',
        f'- Source: [12-shape profile](../{Path(m["profile_source"]).relative_to("problems/muse_decode_layer")}) '
        f'and [source log](../{Path(m["profile_source"]).relative_to("problems/muse_decode_layer").with_suffix(".log")}). '
        f'The adjusted benchmark fields reproduce **{m["baseline_source_speedup"]:.6f}x** exactly. '
        f'The corrected six-load accounting gives **{m["modeled_speedup"]:.6f}x**, a composition model rather than a new score.',
        '- Scale bank: paired receipt `58ae2e2cf8df45ef`; subtract 7.91/9.31/14.43/38.53 us at '
        'M8/16/32/128 from MoE and adjusted layer latency. No paired M1 delta exists, so M1 keeps its measured time. '
        'M256: `97be11d4631b44d6`, masked447.6 minus CTA1g2112 385.2 = **62.4 us**, subtracted once on S/F. '
        'These deltas transfer across router seeds and whole-layer overlap only as a model. '
        'Paired glue/FA3 setting changes from reset to S1 are not assigned extra savings; their '
        'composition needs the S1–S6 profiles. '
        f'[Scale receipt](../../../{m["lab_receipts"][0]["path"]}) · '
        f'[M256 receipt](../../../{m["lab_receipts"][1]["path"]}).',
        '- **Projection role attribution:** the judge merges repeated kernel names. Only a unique custom '
        'GEMV role can be timed directly. Other roles in the same custom/library group are allocated in '
        'proportion to their one-read resource floors and marked **~**. This is an estimate, not recovered '
        'per-role telemetry. Exact group totals and mathematical attribution ranges are retained in JSON.',
        '- **Overlap:** profile rows include PDL dependency waits and, in S6, parallel W13 execution; sums can be above or below the paired '
        'benchmark median. The signed residual is shown separately, never called spare/idle time or '
        'added to a stage. Savings below assume the removed duration was exposed on the critical path; '
        'the profiles alone cannot prove that.',
        '- **Truncation:** six small rows are omitted in each M256 report. The visible MoE and attention '
        'times are lower bounds; their combined omitted mass is in Other. No hidden timing is fabricated. '
        'Other therefore has no defensible single stage floor/target and receives **n/a**, not a zero-cost win.',
        '- **Floors:** ideal cold single read at4.8 TB/s, max with dense BF16 at989 TF/s and legal FP8 QK '
        'at1979 TF/s (QK/PV tensor times add). Intermediate activations may remain on chip. Dense role '
        'shapes/counts and lossless BF16-hi+lo router come from the frozen source. MoE expected touched '
        'experts/routes are inferred from that receipt\'s exact official `speed_of_light_ns`; they are '
        'not its hidden actual routes. Attention uses nominal ctx80000, page1, full key coverage, shared '
        'q8 KV reads; agentic actual lengths vary±7.5%. No replay, launch or measured issue penalty is '
        "charged to the ideal floor. Glue reads five original 8192-wide BF16 vectors and the route stage's 4096-wide "
        'pre-expert vector+bias once, with fused activation IO. Those tiny floors express fusion headroom, '
        'not achievable isolated launch latencies. [Formula source](../../../sim/bound.py), '
        '[hardware rates](../../../sim/hw.py), [official shapes and scoring](../TASK.md).','',
        '## Twelve shapes: times / one-read resource floors (us)','',
        '| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        cells=[]
        for stage in STAGES:
            v=r['stages_us'][stage];floor=r['floors_us'][stage]
            mark='~' if stage in r['estimated_roles'] else '>=' if r['unreported_us'] and stage in ('attention','moe') else ''
            cells.append(f'{mark}{v:.1f} / {floor:.3f}' if floor is not None else f'{v:.1f} / n/a')
        cells.insert(4,f'{sum(r["stages_us"][s] for s in ROLES):.1f} / {sum(r["floors_us"][s] for s in ROLES):.2f}')
        lines.append('| '+shape_label(r)+' | '+' | '.join(cells)+' |')
    lines+=['','S/F=sliding/full; decode M=B, verify M=8B. QKV includes the gate rows; Pool+router includes '
        'the FP32 router\'s BF16 hi+lo products. Attention includes QK/cache preparation, main, combine, '
        'FA3 scheduler and norm/gate. MoE includes expert gather/activation/norm/combine where visible. '
        'Boundaries and route are separate. Split-K reductions belong to their projection group.','',
        '## Overall-score gain per shape/stage at 1.3x floor','',
        'A cell changes that one stage in that one shape, retaining the other eleven. '
        '**All shapes** changes the indicated stage at every shape simultaneously. ~projection cells use '
        'the estimated allocation above; each role\'s source-attribution gain range is shown below.','',
        '| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        cells=[f'{"~" if s in r["estimated_roles"] else ""}+{r["stage_gain_pct"][s]:.2f}%' if s!='other' else 'n/a' for s in STAGES]
        gain=overall_gain(m['rows'],m['loads_us'],lambda x:sum(stage_saving(x,s) for s in ROLES) if x is r else 0.)
        cells.insert(4,f'+{gain:.2f}%');lines.append('| '+shape_label(r)+' | '+' | '.join(cells)+' |')
    cells=[f'+{m["stage_gains"][s]["estimate"]:.2f}%' if s!='other' else 'n/a' for s in STAGES]
    cells.insert(4,f'+{next(x["gain_pct"] for x in m["ranking"] if x["owner"]=="proj GEMV"):.2f}%')
    lines.append('| **All shapes** | '+' | '.join(cells)+' |')
    lines+=['','Gain ranges from unresolved attribution (not confidence intervals):','',
            '| Stage | Central all-shape gain | Source-attribution range |','|---|---:|---:|']
    for s,g in m['stage_gains'].items():lines.append(f'| {NAMES[s]} | +{g["estimate"]:.2f}% | {g["minimum"]:.2f}..{g["maximum"]:.2f}% |')
    lines+=['','Ranges permit a shared kernel group to belong entirely to one of its roles; such extrema '
        'cannot happen for all roles at once. Attention/MoE maxima each assign all M256 omitted mass to '
        'that stage; again these maxima are mutually exclusive. They do not bound PDL critical-path exposure.','',
        '## Latency anchor, corrections and overlap residual','',
        '| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        lines.append('| '+shape_label(r)+' | '+' | '.join(f'{r[k]:.2f}' for k in ('base_adjusted_us','delta_scale_us','delta_m256_us','modeled_adjusted_us','reset_profile_total_after_deltas_us','profile_vs_adjusted_residual_us'))+' |')
    lines+=['','| Load | Adjusted receipt load us | Modeled load us |','|---|---:|---:|']
    for load,value in m['loads_us'].items():lines.append(f'| {load} | {m["unadjusted_loads_us"][load]:.2f} | {value:.2f} |')
    lines+=['','For a selected stage set, `dS/dF = sum(max(0, time - 1.3*floor))` within each shape. '
        'Then `gain% = 100 * (product_load[T / (T - 46*dS - 16*dF)]^(1/6) - 1)`. '
        'Use adjusted `median_ns` anchors above, never the sum of PDL durations. '
        'No regression is assumed if a stage already beats the target.','',
        '## Refresh and checks','',
        '`python problems/muse_decode_layer/dev/levers/build.py` reads local profiles only: chooses frozen '
        'S6 `4c98f12f` first, then S5 `f7064fc0`, S4 `9b2b2a48`, S3 `cadef19a`, S2 `b0805447`, S1 `42f543a0`, otherwise reset. It hashes the immutable submitted '
        'source beside the accepted receipt and verifies the known candidate SHA. '
        'Reset gets both measured corrections; S1 only raw M256; S2–S6 neither. '
        'Use `--receipt problems/muse_decode_layer/profiles/<file>.json` for a named profile. '
        'Run this once when resumed after the09:01 KST receipts land; it makes no judge calls, watches or timers.','',
        '[Machine-readable rows, floors, attribution and validation](levers/current.json) · '
        '[Rebuild and accounting assertions](levers/build.py)','',
        '## Closed inputs to the next integration','',
        'Current negative receipts: FlashInfer full context `8ffbac2d`, persistent chain `87dc944b` '
        '(1.5–6.5x slower), verify split-K WGMMA `19f3a13b` (1.6x slower), and M128 fused producer '
        '`471f6d62` (5.7–6.5x recoded NT1). None supplies a promoted cell. '
        'S3 instead integrates the M256 recoded C1g2112 win `9e8a6a15f95748db` '
        '(365.2 vs raw386.1 us, 12 seeds); S4 adds D32g at M128 and one-LOP3 NT1 '
        'at M1–32 from `dc3fea0a20a349bf`; S5 replaces M128 with H_H from '
        '`770809cf159f41e2` (234.44 vs D32g272.70 us, 36 fixtures). S6 adds M256 mixed_parallel '
        'from8483c0c6 (344.5 vs C1rec366.2 us,12 seeds) and the M1 O/config6 cell (near timing noise). This table retains '
        'its reset + raw-M256 composition basis until the S1–S6 whole-layer profiles land. '
        'Existing source holds remain on `dev/parked/cand_r3_attention_v2.py`, '
        '`dev/parked/cand_r3_v9_glue.py`, both `dev/parked/cand_r3_merge_mega*.py` builds '
        'and the old overlap candidate. Detailed evidence remains in [FINDINGS](../../../docs/FINDINGS.md); '
        '[FOCUS](../FOCUS.md) owns candidate paths and reset allocation.','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--receipt',type=Path);ap.add_argument('--out',type=Path);a=ap.parse_args()
    tag,path,d=choose_source(a.receipt);model=make_model(tag,path,d)
    # Accounting identities, monotonicity and the exact benchmark score are admission checks.
    assert math.isclose(model['modeled_speedup']/model['baseline_source_speedup'],math.prod(model['unadjusted_loads_us'][k]/v for k,v in model['loads_us'].items())**(1/6),rel_tol=1e-12)
    for r in model['rows']:
        assert math.isclose(sum(r['stages_us'][x] for x in ROLES),sum(x['us'] for x in r['projection_groups']),abs_tol=1e-9)
        assert all(g>=0 for g in r['stage_gain_pct'].values())
    scale=model['corrections']['scale_saved_us'];w2=model['corrections']['m256_saved_us']
    for tokens in (1,8,16,32,128,256):
        assert correction_for('S2',tokens,scale,w2)==(0.,0.)
        assert correction_for('S3',tokens,scale,w2)==(0.,0.)
        for integrated_tag in [c[0] for c in CANDIDATES if c[0] not in ('S1','S2','S3')]:assert correction_for(integrated_tag,tokens,scale,w2)==(0.,0.)
        assert correction_for('S1',tokens,scale,w2)==(0.,w2 if tokens==256 else 0.)
    all_moe=model['stage_gains']['moe']['estimate']/100.+1.
    moe_lanes=[x for x in model['ranking'] if x['owner'] in ('small-M MoE','MoE M128','free M256')]
    assert math.isclose(all_moe,math.prod(1.+x['gain_pct']/100. for x in moe_lanes),rel_tol=1e-12)
    model['verification']=dict(shapes=12,loads=6,source_score_reproduced=True,projection_groups_close=True,
                               corrections_applied_once=True,negative_gain_clamped=True,profile_rounding_tolerance_us='0.05 * (visible rows + 1)')
    out=a.out or H.parent/'NEXT_CHECKS.md';out.write_text(render(model));(H/'current.json').write_text(json.dumps(model,indent=2)+'\n')
    print('source',tag,model['job'],'score',model['baseline_source_speedup'],'model',model['modeled_speedup'])
    for i,x in enumerate(model['ranking'],1):print(i,x['owner'],f'{x["gain_pct"]:.4f}%',x['lever'])

if __name__=='__main__':main()
