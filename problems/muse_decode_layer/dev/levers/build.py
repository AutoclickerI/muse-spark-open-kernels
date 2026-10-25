"""Offline, source-pinned 12-shape lever model; refresh from source-verified incumbent profiles."""
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
# MoE engines that fork side-stream branches (M=256: S6 mixed_parallel W13, S7R R_full, S9+ N_full; M=128: S9+ N_full under
# the hh_m128 name): the profile sums both branches' durations, so the overlap (total minus 'GPU busy') comes off the MoE
# stage. A row without a 'GPU busy' line has no overlap to take.
FORKED_MOE=('mixed_parallel_m256','hh_m128')
STAGES=('qkv_gate','o','pool_router','unpooler','attention','moe','boundaries','route','other')
ROLES=STAGES[:4]
NAMES=dict(qkv_gate='QKV+gate',o='O',pool_router='Pool+router',unpooler='Unpool',attention='Attention',moe='MoE',boundaries='Boundaries x2',route='Route+norm',other='Other/unreported')
CANDIDATES=(('S11', 'cand_r3_best7_s11_final', 'acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7'), ('S12', 'cand_r3_best7_s12', 'a92b66193d5cf689c57a1112c6bd166d273c5cb770974c78d57c458a7d300885'), ('S10', 'cand_r3_best7_s10', '6688fad999783d53a4660d85d40ee82862b87a209e03d289e44c95e41a3ffbc1'), ('S9', 'cand_r3_best7_s9', 'f55f616254a88e33781e897de0a8bfbfb82a410d1a03f1da8e8cae0f2de23f6e'), ('S8', 'cand_r3_best7_s8', 'cb84645d1af8b813529a64cfe746dd5846160860deebbe8afd8154007a23c366'), ('S7R', 'cand_r3_best7_s7r', '56c51fa026210357f9865d114e6bf7e609c52049e40b263f589d23ade95a8101'), ('S7', 'cand_r3_best7_s7', 'bd3932d1cc3c0c82d60145772148fd19fc779465dd360bda610d214385401390'), ('S6', 'cand_r3_best7_s6', '4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f'), ('S5', 'cand_r3_best7_s5', 'f7064fc0127e80a955c7dc9aa5c68eb6aefada179c2841c5c58bce8dabd20284'), ('S4', 'cand_r3_best7_s4', '9b2b2a48337aa13ba1daa45eec8a30b566cc3e59c7e1bb317d39695413d3404e'), ('S3', 'cand_r3_best7_scale_recoded_m256', 'cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7'), ('S2', 'cand_r3_best7_scale_w2fix', 'b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337'), ('S1', 'cand_r3_best7_scale', '42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7'))

# Default follows the incumbent, not the newest experiment. Explicit --receipt admits S12.
LAB_SOURCES = {
    '863f502c62e54d7b': ('S11', '04a4859d759f7d29afb722d01e2c6bcd6e7c218a03f15ef0f7fe492e71dce6b1'),
    '937801feb77b4302': ('S12', 'c64aed260534aa9040f1a0aeb07b597d545392e3fab535349ef6efdb1fa888e2'),
}
TARGETS = (1.5, 1.3, 1.15)
STREAM_BPS = 4.58e12
STREAM_INTERCEPT_US = 2.36


def source_identity(d):
    paths = sorted((R/'runs/muse_decode_layer').glob('*_'+d['job_id']+'.json'))
    if len(paths) != 1:
        raise ValueError('Exactly one accepted source receipt is required')
    receipt = json.loads(paths[0].read_text())
    source = paths[0].with_suffix('.py').read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    if 'sha256' in receipt:
        assert receipt['sha256'] == sha
    timed = source
    if d['job_id'] in LAB_SOURCES:
        tag, expected = LAB_SOURCES[d['job_id']]
        assert sha == expected, ('lab wrapper SHA', sha, expected)
        # h200_lab appends one newline before this marker. Verify the entire base,
        # not just the wrapper comment's claimed tag or the live candidate path.
        prefix, _ = source.split(b'\n# ---- h200_lab wrapper:', 1)
        assert prefix.endswith(b'\n')
        timed = prefix[:-1]
        assert hashlib.sha256(timed).hexdigest() == next(c[2] for c in CANDIDATES if c[0] == tag)
    timed_sha = hashlib.sha256(timed).hexdigest()
    candidates = (('reset', '', timed_sha),) if d['job_id']==RESET else CANDIDATES
    for tag, _, expected in candidates:
        if timed_sha == expected:
            return dict(tag=tag, submitted_sha256=sha, timed_sha256=timed_sha,
                        receipt=str(paths[0].relative_to(R)))
    raise ValueError('Unsupported source SHA: identify integrated deltas before using it')


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
    return 'reset' if d['job_id'] == RESET else source_identity(d)['tag']


def choose_source(explicit):
    if explicit:
        p=explicit.resolve();d=profile_file(p)
        return source_tag(d),p,d
    for job, (tag, _) in LAB_SOURCES.items():
        if tag == 'S11':
            p = next((P.parent/'lab/profiles').glob('*.'+job+'.json'))
            d = profile_file(p)
            assert source_tag(d) == tag
            return tag, p, d
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


def floor_data(case,bench,tag):
    t=case['batch']*case.get('q_len',1)
    works={n:bound.projection_work(dict(m=t,layer=n),activation_factor=0) for n in ('qkv_gate','o','pooler','router','unpooler')}
    works['pool_router']=works['pooler']+works['router']
    coded = {}
    # S12's t=1 gate-split path calls qkv3 through gemv(), bypassing CD_ROWS.
    # Only pool_router is coded in these profiles (one cdg::coded_gemv call).
    if tag == 'S12' and t == 1:
        n, k = 4608, 8192  # pooler + BF16 hi/lo router
        # Payload 12 bits/weight + scale per 16 rows + offset per 16x64 block.
        # Escape count is not in the truncated log: zero escapes is a lower bound.
        coded['pool_router'] = n*k*1.5 + (n//16)*4 + ((n//16)*(k//64)+1)*4
        w = works['pool_router']
        works['pool_router'] = bound.Work(coded['pool_router'], w.bf16_flops)
    # Cold, single-read logical weights. BF16 hi+lo router retains its four bytes.
    floors={n:max(works[n].nbytes/hw.HBM_BPS,works[n].bf16_flops/hw.BF16_FLOPS)*1e6 for n in ROLES}
    touched,routes=bound.inferred_routing('muse_decode_layer',case,bench['speed_of_light_ns']/1e3)
    a=bound.attention_work(case['batch'],case.get('q_len',1),case['ctx'],case['layer'],external_io=False)
    m=bound.nvfp4_expert_work(touched,routes,t,external_io=False)
    resource=lambda w:max(w.nbytes/hw.HBM_BPS,(w.bf16_flops/hw.BF16_FLOPS+w.fp8_flops/hw.FP8_FLOPS))*1e6
    floors.update(attention=resource(a),moe=resource(m),boundaries=(5*8192*2)/hw.HBM_BPS*1e6,
                  route=(4096*2+256*4)/hw.HBM_BPS*1e6,other=None)
    return floors,dict(touched=touched,routes=routes,context=case['ctx'],projection_bytes={n:works[n].nbytes for n in ROLES},
                       attention_bytes=a.nbytes,moe_bytes=m.nbytes,coded_bytes_lower_bound=coded)


def parse_profile(report,plan,tokens,floors,tag,case):
    values={k:0. for k in STAGES};groups=[];kernels=[]
    pattern=r'^(.*?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)\s*$'
    for line in report.splitlines():
        match=re.match(pattern,line)
        if not match:continue
        name,us,calls=match[1],float(match[2]),int(match[3]);kernels.append(dict(name=name,us=us,calls=calls))
    assert kernels
    modern = tag in ('S11', 'S12')
    custom=[];fixed=[];vdsk=[]
    for role,choice in plan['projection'].items():
        normalized='unpooler' if role=='unpooler_slice' else role
        if choice=='custom' and tokens<=64 and (normalized!='unpooler' or modern and tokens<=32):custom.append(normalized)
        elif choice=='custom' and tokens<=8 and normalized=='unpooler':custom.append(normalized)
        elif isinstance(choice,(tuple,list)) and choice and choice[0]=='vdsk':vdsk.append(normalized)
        elif isinstance(choice,(tuple,list)) and tuple(choice)==('stream',6,1):
            assert (tokens,normalized)==(1,'o'),(tokens,normalized,choice)
            fixed.append(normalized)
    split = modern and case.get('q_len', 1) == 1 and (case['batch'], case['layer']=='sliding') in {(1, True), (1, False), (16, True), (32, False)}
    balanced = [r for r in custom if modern and ((r in ('qkv_gate', 'pool_router')) or (r=='unpooler' and tokens in (16,32)))]
    coded = ['pool_router'] if tag == 'S12' and tokens == 1 else []
    balanced = [r for r in balanced if r not in coded]
    plain = [r for r in custom if r not in balanced and r not in coded]
    groups_by={}
    def group(roles, u, name, fractions=None):
        assert roles, (plan, name)
        key = (tuple(roles), tuple((fractions or {}).items()))
        g = groups_by.setdefault(key, dict(roles=tuple(roles), us=0., names=[], floor_fractions=fractions or {}))
        g['us'] += u
        g['names'].append(name)
    for k in kernels:
        n,u=k['name'],k['us']
        if '_gate_split_kernel' in n:
            assert split
            group(('qkv_gate',),u,n,{'qkv_gate':4/9})
        elif '_gb_kernel' in n:
            group(balanced,u,n,{'qkv_gate':5/9} if split else None)
        elif 'cdg::coded_gemv<' in n:
            group(coded,u,n)
        elif '_gemv_kernel' in n:
            group(plain,u,n)
        elif '_vdsk_gemm' in n:
            group(vdsk,u,n)
        elif 'gf::direct<' in n:
            group(fixed,u,n)
        elif any(x in n for x in ('nvjet_','cublasLt::splitKreduce')):
            group([r for r in ROLES if r not in custom and r not in fixed and r not in vdsk],u,n)
        elif any(x in n for x in MOE_KERNELS):values['moe']+=u
        elif any(x in n for x in ('attn_main','attn_combine','flash::','vq8_main','vq8_combine','_qk_prep_kernel','_norm_gate_kernel','_attn_sumcast_kernel')):values['attention']+=u
        elif any(x in n for x in ('_boundary_','_best7_old_boundary','boundary_cl<')):values['boundaries']+=u
        elif any(x in n for x in ('_route_','_best7_old_route','route_cl<')):values['route']+=u
        else:values['other']+=u
    assert sum(k['calls'] for k in kernels if '_gemv_kernel' in k['name'])==len(plain),(tokens,plain,kernels)
    assert sum(k['calls'] for k in kernels if '_gb_kernel' in k['name']) == len(balanced)
    assert sum(k['calls'] for k in kernels if 'cdg::coded_gemv<' in k['name']) == len(coded)
    assert sum(k['calls'] for k in kernels if '_gate_split_kernel' in k['name']) == int(split)
    assert sum(k['calls'] for k in kernels if 'gf::direct<' in k['name'])==len(fixed),(tokens,fixed,kernels)
    assert sum(k['calls'] for k in kernels if '_vdsk_gemm' in k['name'])==len(vdsk),(tokens,vdsk,kernels)
    assert sum(k['calls'] for k in kernels if 'nvjet_' in k['name'])==4-len(custom)-len(fixed)-len(vdsk),(tokens,custom,fixed,vdsk,kernels)
    role_bounds={r:[0.,0.] for r in ROLES};estimated=[]
    for g in groups_by.values():
        roles=g['roles']
        allocation={r:floors[r]*g['floor_fractions'].get(r,1.) for r in roles}
        denom=sum(allocation.values())
        for role in roles:
            values[role]+=g['us']*allocation[role]/denom
            role_bounds[role][0]+=g['us'] if len(roles)==1 else 0.
            role_bounds[role][1]+=g['us']
            if len(roles)>1:estimated.append(role)
        groups.append(dict(**g,allocation='exact role' if len(roles)==1 else 'modeled proportional resource floor'))
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


def stage_saving(row,stage,kind='estimate',target=1.3):
    if stage=='other':return 0.
    time=row['stages_us'][stage]
    if kind in ('min','max') and stage in ROLES:time=row['role_bounds_us'][stage][kind=='max']
    if kind=='max' and stage in ('moe','attention'):time+=row['unreported_us']
    return max(0.,time-target*row['floors_us'][stage])


def make_model(tag,path,d):
    scale,w2,lab_sources=deltas();plans=plan_for(d,path);rows=[];loads=defaultdict(float);base_loads=defaultdict(float);baseline=defaultdict(float)
    for p,b in zip(d['profiles'],d['benchmarks']):
        assert p['case']==b['case'];c=dict(p['case']);t=c['batch']*c.get('q_len',1)
        key=c['batch'],c.get('q_len',1),2048 if c['layer']=='sliding' else 0
        f,work=floor_data(c,b,tag)
        v,groups,ranges,estimated,missing,total,kernels,overlap=parse_profile(p['report'],plans[key],t,f,tag,c)
        delta_scale,delta_w2=correction_for(tag,t,scale,w2)
        delta=delta_scale+delta_w2
        v['moe']-=delta
        assert v['moe']>0
        row=dict(index=p['index'],case=c,tokens=t,stages_us=v,floors_us=f,work=work,role_bounds_us=ranges,
                 estimated_roles=estimated,projection_groups=groups,unreported_us=missing,moe_branch_overlap_us=overlap,profile_total_us=total,
                 reset_profile_total_after_deltas_us=total-delta,base_adjusted_us=b['median_ns']/1000.,modeled_adjusted_us=b['median_ns']/1000.-delta,
                 delta_scale_us=delta_scale,delta_m256_us=delta_w2,base_moe_us=v['moe']+delta,benchmark_baseline_us=b['baseline_ns']/1000.,
                 kernels=kernels,plan=plans[key])
        row['time_floor_ratio']={s:v[s]/f[s] for s in STAGES if f[s] is not None}
        stage_bytes={**work['projection_bytes'],'attention':work['attention_bytes'],'moe':work['moe_bytes']}
        row['stream_roof_us']={s:max(f[s],STREAM_INTERCEPT_US+n/STREAM_BPS*1e6) for s,n in stage_bytes.items()}
        row['excess_1_3_us']={s:stage_saving(row,s) for s in STAGES if s!='other'}
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
        gains={str(f):overall_gain(rows,loads,lambda r,f=f:sum(stage_saving(r,s,target=f) for s in stages) if predicate(r) else 0.) for f in TARGETS}
        ranking.append(dict(lever=name,owner=owner,gain_pct=gains['1.3'],gains_pct=gains,
                            gain_max_unreported_pct=overall_gain(rows,loads,lambda r:save(r)+(r['unreported_us'] if predicate(r) and any(s in ('moe','attention') for s in stages) else 0.)),
                            saved_weighted_us=sum(r['case']['weight']*save(r) for r in rows)))
    ranking.sort(key=lambda x:x['gain_pct'],reverse=True)
    excess=[]
    for stage in STAGES:
        if stage=='other':continue
        per_load={load:sum(r['case']['weight']*stage_saving(r,stage) for r in rows if r['case']['load']==load) for load in loads}
        excess.append(dict(stage=stage,weighted_us=sum(per_load.values()),per_load_us=per_load))
    excess.sort(key=lambda x:x['weighted_us'],reverse=True)
    worst=sorted([dict(shape=shape_label(r),stage=s,time_us=r['stages_us'][s],floor_us=r['floors_us'][s],excess_us=stage_saving(r,s)) for r in rows for s in STAGES if s!='other'],key=lambda x:x['excess_us'],reverse=True)
    return dict(mode=tag,source_identity=source_identity(d),
                integrated_features=(['scale bank', 'M256 recoding', 'small-M flags', 'N_swap', 'BF16 front', 'balanced split-K GEMVs', 'gate split', 'late glue', 'LI_swap N_full', 'vq8 verify 16/32 full'] + (['F4 finalizer', 'wide banks', 'coded t=1 pool_router (qkv3 bypasses CD_ROWS)'] if tag=='S12' else [])) if tag in ('S11','S12') else [],
                targets=TARGETS,stream_model=dict(intercept_us=STREAM_INTERCEPT_US,bytes_per_second=STREAM_BPS,job='7b6c96b5'),
                excess_1_3_by_stage=excess,worst_shape_stages=worst,profile_source=str(path.relative_to(R)),profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),profile_log_sha256=hashlib.sha256(path.with_suffix('.log').read_bytes()).hexdigest(),job=d['job_id'],
                baseline_source_speedup=source_score,modeled_speedup=math.prod(baseline[k]/loads[k] for k in loads)**(1/6),
                loads_us=loads,baseline_loads_us=baseline,unadjusted_loads_us=base_loads,stage_gains=stage_gains,ranking=ranking,rows=rows,
                corrections=dict(scale_job=SCALE,scale_saved_us=scale,m256_job=W2,m256_saved_us=w2,m1='no matched M1 delta; no invented gain'),
                model_sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(),R/'sim/hw.py',R/'sim/bound.py',R/'sim/muse.py',R/'sim/problems.py')},
                lab_receipts=[dict(path=str(p.relative_to(R)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,_ in lab_sources])


def shape_label(r):return r['case']['load']+' '+('S' if r['case']['layer']=='sliding' else 'F')


def render(m):
    profile=Path(m['profile_source']).relative_to('problems/muse_decode_layer')
    identity=m['source_identity']
    lines=['# Decode P4 — current stage levers','',
           '## Ranking: one lane at 1.3x its floor','',
           f'**Basis:** H200 profile `{m["job"]}`; **{m["mode"]}** timed source '
           f'`{identity["timed_sha256"][:8]}`. '+('All integrated changes are already measured; no extra deltas.' if m['mode'] in ('S11','S12') else 'Historical source; only its missing corrections apply.'),'',
           '| Rank | Lever | Owning lane | Gain at 1.5x | Gain at 1.3x | Gain at 1.15x |',
           '|---|---|---|---:|---:|---:|']
    for i,row in enumerate(m['ranking'],1):
        gains=' | '.join(f'+{row["gains_pct"][str(f)]:.2f}%' for f in TARGETS)
        lines.append(f'| {i} | {row["lever"]} | {row["owner"]} | {gains} |')
    lines += ['',
        'Ranked at 1.3x. Each row caps only that lane’s stages and shapes; gains are independent '
        'counterfactuals and must not be added. These are optimistic profile-duration headrooms, '
        'not measured wins. Shared projection attribution does not affect total projection time, '
        'but per-role clamping can affect its gain. PDL waits and routing seed variation can change the ordering.','',
        '## What is measured and what is modeled','',
        f'- Source: [12-shape profile](../{profile}), [source log](../{profile.with_suffix(".log")}), '
        f'[receipt](../../../{identity["receipt"]}). The immutable submitted wrapper '
        f'`{identity["submitted_sha256"][:8]}` contains a byte-verified base '
        f'`{identity["timed_sha256"][:8]}`. Per-load sums reproduce **{m["baseline_source_speedup"]:.9f}x**; '
        f'the model is **{m["modeled_speedup"]:.9f}x**. This is the host profile’s score, not a new submission.',
        ('- **Integrated:** '+', '.join(m['integrated_features'])+'. Scale-bank and raw-M256 corrections are '
        'zero at every shape. No balanced-GEMV, gate, vq8, LI_swap, F4 or coded savings are subtracted from '
        'measured times.' if m['mode'] in ('S11','S12') else '- **Historical corrections:** reset receives scale-bank and raw-M256 deltas; S1 receives only raw-M256; S2 onward receives neither. See the correction columns below.'),
        '- **Projection attribution:** repeated names are merged. Balanced `_gb_kernel`, plain `_gemv_kernel`, '
        'coded `cdg::coded_gemv`, library and split-K groups are kept separate. Shared groups are split by '
        'their one-read floors and marked **~**. With gate split, the balanced QKV3 share uses 5/9 of the '
        'QKV+gate bytes; `_gate_split_kernel` contributes its full duration to QKV+gate (the other 4/9). '
        'Its gate application is fused there; attention retains sum/cast, main and combine.',
        '- **Overlap:** profile durations include PDL waits. For forked N_full, the report’s total-minus-GPU-busy '
        'overlap is deducted from MoE, following the existing attribution convention; it can also contain '
        'PDL overlap outside MoE. It is not a recovered MoE critical-path measurement. Other shapes retain '
        'their PDL durations. Signed layer-minus-profile residuals remain explicit. No residual is labeled '
        'idle time or assigned to a stage.',
        '- **Truncation:** omitted rows remain in Other. Visible attention and MoE are lower bounds where '
        'rows are omitted; upper attribution bounds assign all omitted mass to either one, never both. '
        'Other has no defensible floor or gain.',
        '- **Floors:** `sim/bound.py` work counts, ideal one-read bytes at 4.8 TB/s, BF16 at 989 TF/s, '
        'FP8 QK at 1979 TF/s; QK/PV tensor times add. These resource floors exclude fixed launch/issue '
        'costs and intermediate activation IO. Routing expectations are inferred from the profile’s official '
        '`speed_of_light_ns`, not hidden actual routes. Attention uses nominal context 80000 and all keys; '
        'actual agentic lengths vary. Glue floors count fused compulsory vector IO and express fusion '
        'headroom, not isolated-kernel latency.',
        '- **Measured stream roof:** **2.36 us + bytes / 4.58 TB/s**, from `7b6c96b5` '
        '([FINDINGS](../../../docs/FINDINGS.md), cold read). JSON keeps this separately for each data stage '
        '(max with its resource floor); the 1.5/1.3/1.15 targets use the ideal floor. The fixed intercept '
        'is not another replay correction or a guaranteed overlap saving. A split stage can pay multiple '
        'intercepts. [Work formulas](../../../sim/bound.py) · [rates](../../../sim/hw.py) · [scoring](../TASK.md).']
    if m['mode']=='S12':
        lines += ['- **Coded floor:** only pool_router is coded at decode 1: gate split directly calls '
                  'balanced QKV3 and bypasses the nominal qkv CD_ROWS entry. Its floor counts 12-bit payload, '
                  'scale per 16 rows and offsets per 16x64 block. Escape bytes are unknown in the truncated '
                  'log and excluded, so this is a lower bound, not an exact encoded footprint.']
    lines += ['', '## Twelve shapes: times / one-read resource floors (us)', '',
              'Cells show **time / floor (ratio)**. **†** marks time above 1.3x floor; the excess tables '
              'below rank absolute headroom, since tiny glue floors create very large ratios.', '',
              '| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        cells=[]
        for stage in STAGES:
            v=r['stages_us'][stage];floor=r['floors_us'][stage]
            mark='~' if stage in r['estimated_roles'] else '>=' if r['unreported_us'] and stage in ('attention','moe') else ''
            cells.append(f'{mark}{v:.1f} / {floor:.3f} ({v/floor:.2f}x){"†" if v>1.3*floor else ""}' if floor is not None else f'{v:.1f} / n/a')
        v=sum(r['stages_us'][s] for s in ROLES);f=sum(r['floors_us'][s] for s in ROLES)
        cells.insert(4,f'{v:.1f} / {f:.2f} ({v/f:.2f}x)')
        lines.append('| '+shape_label(r)+' | '+' | '.join(cells)+' |')
    lines += ['', 'S/F=sliding/full; decode M=B, verify M=8B. QKV includes gate, Pool+router includes '
              'the FP32 router’s BF16 hi+lo products. Attention includes prep, main, combine, scheduler '
              'and norm/gate except the fused gate split charged above. N_full’s LI_swap heavy and NT1 '
              'branches, metadata and finalizer belong to MoE. Boundaries and route remain separate.', '',
              '### Stages furthest above 1.3x, across the six loads', '',
              'Excess per load is `46*max(0,tS-1.3*fS) + 16*max(0,tF-1.3*fF)` in us. '
              'The sum gives each of the six loads one step; score ranking instead uses relative load '
              'savings and a geomean. These are stage-duration estimates subject to the overlap caveat above.', '',
              '| Stage | decode 1 | decode 16 | decode 32 | verify 1 | verify 16 | verify 32 | Six-load sum us |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in m['excess_1_3_by_stage']:
        lines.append('| '+NAMES[x['stage']]+' | '+' | '.join(f'{v:.1f}' for v in x['per_load_us'].values())+f' | **{x["weighted_us"]:.1f}** |')
    lines += ['', 'Largest individual shape/stage excesses (us per layer):', '',
              '| Shape | Stage | Time us | Floor us | Above 1.3x us |', '|---|---|---:|---:|---:|']
    for x in m['worst_shape_stages'][:3]:
        lines.append(f'| {x["shape"]} | {NAMES[x["stage"]]} | {x["time_us"]:.2f} | {x["floor_us"]:.3f} | {x["excess_us"]:.2f} |')
    lines += ['', '## Overall-score gain per shape/stage at 1.3x floor', '',
              'Each cell changes one stage in one shape. All shapes changes that stage everywhere. '
              '~ marks modeled projection allocation; attribution bounds follow.', '',
              '| Shape | QKV+gate | O | Pool+router | Unpool | Proj sum | Attention | MoE | Boundaries x2 | Route+norm | Other |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        cells=[f'{"~" if s in r["estimated_roles"] else ""}+{r["stage_gain_pct"][s]:.2f}%' if s!='other' else 'n/a' for s in STAGES]
        gain=overall_gain(m['rows'],m['loads_us'],lambda x:sum(stage_saving(x,s) for s in ROLES) if x is r else 0.)
        cells.insert(4,f'+{gain:.2f}%');lines.append('| '+shape_label(r)+' | '+' | '.join(cells)+' |')
    cells=[f'+{m["stage_gains"][s]["estimate"]:.2f}%' if s!='other' else 'n/a' for s in STAGES]
    cells.insert(4,f'+{next(x["gain_pct"] for x in m["ranking"] if x["owner"]=="proj GEMV"):.2f}%')
    lines.append('| **All shapes** | '+' | '.join(cells)+' |')
    lines += ['', 'Gain ranges from unresolved attribution (not confidence intervals):', '',
              '| Stage | Central all-shape gain | Source-attribution range |', '|---|---:|---:|']
    for stage,g in m['stage_gains'].items():
        lines.append(f'| {NAMES[stage]} | +{g["estimate"]:.2f}% | {g["minimum"]:.2f}..{g["maximum"]:.2f}% |')
    lines += ['', 'Shared-group extrema cannot occur for all roles at once. Attention/MoE maxima '
              'each consume all omitted mass and are mutually exclusive. These ranges do not bound '
              'critical-path exposure.', '', '## Latency anchor, corrections and overlap residual', '',
              '| Shape | Receipt layer | Scale saving | M256 saving | Model layer | Profile sum after deltas | Layer minus profile |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for r in m['rows']:
        lines.append('| '+shape_label(r)+' | '+' | '.join(f'{r[k]:.2f}' for k in ('base_adjusted_us','delta_scale_us','delta_m256_us','modeled_adjusted_us','reset_profile_total_after_deltas_us','profile_vs_adjusted_residual_us'))+' |')
    lines += ['', '| Load | SGLang load us | Adjusted receipt load us | Modeled load us | SGLang/model |', '|---|---:|---:|---:|---:|']
    for load,value in m['loads_us'].items():
        lines.append(f'| {load} | {m["baseline_loads_us"][load]:.6f} | {m["unadjusted_loads_us"][load]:.6f} | {value:.6f} | {m["baseline_loads_us"][load]/value:.9f} |')
    lines += ['', 'For target `f`, `dS/dF = sum(max(0,time-f*floor))` within each shape. '
              '`gain% = 100*(product_load[T/(T-46*dS-16*dF)]^(1/6)-1)`. Use adjusted benchmark '
              '`median_ns` anchors, never summed PDL durations. A stage already below target is unchanged.', '',
              '## Refresh and checks', '',
              '`python3 problems/muse_decode_layer/dev/levers/build.py` selects the incumbent S11 final lab '
              'profile by default. `--receipt problems/muse_decode_layer/lab/profiles/lab_decode_s12ab.937801feb77b4302.json` '
              'selects S12 revision a92b6619 explicitly. Both wrapper SHA and the complete embedded timed source '
              'SHA are verified. Other submitted sources must match a registered candidate SHA. Historical '
              'reset gets scale and raw-M256 corrections, S1 gets only raw-M256, S2 and later get neither.', '',
              'The builder verifies twelve shapes, six loads, exact score reproduction, stage accounting, '
              'kernel call counts, projection groups and correction ownership. '
              '`python3 problems/muse_decode_layer/dev/levers/check.py` checks both new profiles and '
              'source rejection, attribution, stream and target monotonicity. No judge calls.', '',
              '[Machine-readable evidence](levers/current.json) · [Builder](levers/build.py)', '',
              '## Inputs to the next integration', '',
              'S11 final is the incumbent. S12 a92b6619 is supported as an experiment: its paired '
              '`937801fe` leave-one-out finds coded-off +3.9% at decode 1 (A:A +1.1%), F4-off +0.60/+0.73% '
              'at verify 16/32, and wide-off -0.04/+0.15% (noise). Cross-job host scores do not rank '
              'candidates. Future promotions need paired whole-layer measurements; these modeled caps '
              'are priorities for the lanes. [FINDINGS](../../../docs/FINDINGS.md).', '']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--receipt',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--json-out',type=Path);a=ap.parse_args()
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
    out=a.out or H.parent/'NEXT_CHECKS.md';out.write_text(render(model));(a.json_out or (a.out.with_suffix('.json') if a.out else H/'current.json')).write_text(json.dumps(model,indent=2)+'\n')
    print('source',tag,model['job'],'score',model['baseline_source_speedup'],'model',model['modeled_speedup'])
    for i,x in enumerate(model['ranking'],1):print(i,x['owner'],f'{x["gain_pct"]:.4f}%',x['lever'])

if __name__=='__main__':main()
