"""Receipt-grounded decomposition and explicitly conditional six-load cost model."""
import hashlib, json, math, re, statistics, sys
from pathlib import Path
D=Path(__file__).resolve().parent
R=D.parents[5]
sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines
inputs={}
def read(path):
    p=R/path
    inputs[path]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
dep=read(str((D/'results.json').relative_to(R)))
counts=read(str((D/'sizing_counts.json').relative_to(R)))
assert dep['measured'] and dep['source_audit']['verified'] and not dep['invalid']
assert not dep['seed_level_record']['checks']['failures']
assert len(dep['summary'])==15 and all(not v['point_nominee'] for v in dep['summary'])
stream=read('runs/muse_moe_w4a16/20261025T075730Z_check_360810b110a44d36.json')
phase=read('runs/muse_prefill_layer/20261025T021729Z_submit_c768a8e361f54977.json')
assert stream['result']['passed'] and phase['result']['passed']
text='\n'.join(result_lines(stream['result']['log_tail']))
assert 'END seeds=16 A_worst_task_L2=0.00238 checksums=C+D_PASS' in text
aggs={}
for m,arm,us,ratio,se in re.findall(r'AGG M(8|16|32) (A|B|C|D4) node_us=([\d.]+).*?paired_r=([\d.]+) se=([\d.]+)',text):
    aggs[(int(m),arm)]=dict(node_us=float(us),paired_ratio=float(ratio),SE=float(se))
assert len(aggs)==12
clocks=read('problems/muse_decode_layer/dev/nt1_timeline/structural/overlap/clock_results.json')
assert clocks['source_audit']['verified']
table=[]
for m in (8,16,32):
    o=next(v for v in dep['oracle'] if v['M']==m)
    c=next(v for v in clocks['records'] if v['M']==m and v['kind']=='BASE')
    steady=[(b-a)/1000 for a,b in zip(c['first8_math_begin_ns'][1:],c['first8_math_end_ns'][1:])]
    table.append(dict(M=m,S19_routed_stage_us=o['baseline_us'],S19_dependency_removed_us=o['saved_us'],
        S19_dependency_ratio=dict(ratio=o['ratio'],SE=o['SE'],CI95=o['CI95'],n=16),
        pure_stream_historical_measured_us=aggs[(m,'D4')]['node_us'],
        loads_only_historical_measured_us=aggs[(m,'C')]['node_us'],
        historical_correct_node_us=aggs[(m,'A')]['node_us'],
        historical_math_activation_wait_epilogue_contrast_us=aggs[(m,'A')]['node_us']-aggs[(m,'C')]['node_us'],
        historical_load_issue_schedule_contrast_us=aggs[(m,'C')]['node_us']-aggs[(m,'D4')]['node_us'],
        S19_unassigned_after_stream_and_wait_model_us=o['baseline_us']-aggs[(m,'D4')]['node_us']-o['saved_us']['mean'],
        W13_start_issue_to_math_sample_us=(c['first8_math_begin_ns'][0]-c['first8_issue_ns'][0])/1000,
        W2_first_tile_us=None,
        last_tile_math_model_each_phase_us=statistics.median(steady),
        last_tile_math_sample_range_us=[min(steady),max(steady)],
        accumulated_epilogue_warp0_residence_us=c['warp_rows'][0][6]/1000,
        epilogue_per_average_tile_model_us=c['warp_rows'][0][6]/1000/c['warp_rows'][0][7],
        clock_observer_ratio=c['observer_ratio'],
        phase_split_status='No per-phase last-tile/epilogue observations retained;steady W13 samples proxy last-tile math only;not an additive normal-kernel split',
        exclusive_compute_not_hidden_us=None))
# Score scale uses official S18 46/16-weighted latencies transported by the S19/S18
# same-input load ratio. It is a planning model,not a fresh S19 wall-time measurement.
s18=read('runs/muse_decode_layer/20261026T050310Z_submit_28b5c189e7b5469c.json')
s19=read('problems/muse_decode_layer/dev/s19/batches/22f25954e6ee84ab/report.json')
assert s18['result']['passed'] and s19['source_sha256']==dep['seed_level_record']['identities']['base_S19']
assert s19['incumbent_sha256'].startswith('95df42fe')
latencies={}
for load in ('verify 1','decode 16','decode 32'):
    rows=[v for v in s18['result']['records'] if v.get('event')=='benchmark' and v['case']['load']==load]
    assert len(rows)==2 and sum(v['case']['weight'] for v in rows)==62
    t=sum(v['case']['weight']*v['median_ns']/1000 for v in rows)/62
    ratio=statistics.mean(v['INC'] for v in s19['raw']['loads'][load]['samples'])
    latencies[load]=dict(S18_measured_weighted_us=t,S18_over_S19_paired_mean=ratio,S19_modeled_weighted_us=t/ratio)
ts=[v['S19_modeled_weighted_us'] for v in latencies.values()]
saved=[v['S19_dependency_removed_us']['mean'] for v in table]
def Rcost(cost):
    # Per-shape choice can retain S19 at any losing M. Only three of six loads change.
    return math.exp(sum(-math.log1p(-max(0,a-b)/t) for a,b,t in zip(saved,cost,ts))/6)
lo,hi=0.,max(saved)
for _ in range(64):
    mid=(lo+hi)/2
    if Rcost([mid]*3)>1.01:lo=mid
    else:hi=mid
budget=lo
profiles={(v['M'],v['arm']):v for v in dep['profile_summary']}
options=[]
for v in counts['rows']:
    C=v['cluster_blocks'];S=v['slice']
    # This is an intentionally favorable count-scaled proxy. It is NOT measured DSM
    # throughput,an upper/lower performance bound,or independent atomic latency.
    proxy=[next(v['main_minus_NORED']['mean'] for v in dep['output_cost'] if v['M']==m and v['arm']=='A128D')*C/32 for m in (8,16,32)]
    work=[]
    for m in (8,16,32):
        w=next(x for x in dep['work'] if x['M']==m and x['arm']=='A128D')
        work.append(dict(M=m,clusters=w['mean_token_blocks'],useful_CTAs=C*w['mean_token_blocks'],
                         useful_CTAs_per_SM_upper=C*w['mean_token_blocks']/132,
                         FP32_DSM_reductions=C*w['mean_local_routes']*4096,
                         same_copy_W13_tiles_per_CTA=S//8,W2_full_tile_equivalents_per_CTA=32*S//512,
                         total_TMA_copy_ratio_vs_S19=(512+(4096//S)*32)/768))
    options.append(dict(kind=f'owner_sharded_DSM_C{C}_S{S}',budget=v,work=work,
                        added_output_path_proxy_us=proxy,modeled_R_before_sync_underfill_finish=Rcost(proxy),
                        sensitivity_R_if_proxy_halves_and_sync_is_1us=Rcost([a/2+1 for a in proxy]),
                        sensitivity_R_if_proxy_halves_and_sync_is_half_us=Rcost([a/2+.5 for a in proxy]),
                        actual_DSM_cost_CI=None,cluster_occupancy_measured=False,
                        verdict='Not admitted:count-scaled point proxy<1% before extra costs;faster DSM/changed math scheduling can reopen;not an infeasibility proof'))
finish=[]
for m in (8,16,32):
    a=profiles[(m,'A128D')];b=profiles[(m,'BASE')]
    finish.append(dict(M=m,atomic_main_us=a['main']['mean'],BASE_main_us=b['main']['mean'],
                       finish_zero_norm_combine_us=sum(a[k]['mean'] for k in ('zero','norm','combine')),
                       excess_main_even_if_all_finish_free_us=a['main']['mean']-b['main']['mean'],
                       status='First4seed CUPTI means;optimistic zero-finish counterfactual,not a timed valid arm'))
gap=read('problems/muse_decode_layer/dev/p12_components/leader_gap.json')
leader=next(v for v in gap['ranking'] if v['role']=='moe')
out=dict(status='Dependency/DSM forms not admitted by this sizing; lane active under matched S19 ladder',inputs_sha256=inputs,
         decomposition=table,
         historical_phase_M32=dict(receipt='c768a8e361f54977',W13_us=36.05,W2_ready_us=11.58,W2_rest_us=17.57,grid_us=4.98,combine_us=1.17,scope='Old instrumented critical CTA;not S19 or additive whole-stage savings'),
         score_model=dict(loads=latencies,loads_changed=3,total_loads=6,full_credit_no_overlap_assumption=True,
                          free_wait_removal_R=Rcost([0]*3),common_added_cost_us_for_R1p01=budget,
                          status='Model from cross-run latencies;not G0 confirmation,not a universal DSM ceiling'),
         options=options,
         tile_gather_DSM=dict(complete_partial_bytes_per_CTA=8*4096*4,raw_live_bytes=2*36864,
                              minimum_shared_bytes=8*4096*4+2*36864,CTAs_per_SM=1,
                              streamed_alternative=dict(output_chunks=32,minimum_buffer_reuse_cluster_barriers=64,
                                                        total_common_budget_us=budget,ns_per_barrier_if_all_other_new_work_free=budget*1000/64,
                                                        measured_barrier_latency=None),
                              verdict='Full gather fails2CTA;streaming gather has no calibrated cost under the budget;unimplemented/unmeasured'),
         fused_finish=finish,
         leader=dict(snapshot=gap['updated_utc'],our_base=gap['base_sha256'],ratio=leader['time_ratio_to_component_leader'],
                     implied_component_time_reduction=1-1/leader['time_ratio_to_component_leader'],
                     leader_job=leader['leader_job'],our_receipt=leader['receipt'],per_shape_available=leader['leader_per_shape_available'],
                     interpretation='Aggregate S15 component gap,not a per-M S19 result,not a stream ceiling;could be heavy-shape or other algorithm work'),
         reopening=['Coarse-cluster actual occupancy2CTA/no spills with enough useful expert clusters',
                    'Matched DSM+coarse scheduling+finish cost below the per-load1% budget(about2.13us common cost),or a measured additional compute/load-issue saving',
                    'New exact dequant/layout/stream schedule with source/resource proof and a credible>1%six-load model',
                    'Leader per-shape/public implementation evidence locating a recoverable small-M mechanism;aggregate1.26alone is insufficient'],
         new_arm_ratios_CI=None,new_H200_jobs=[],P4_new_nominees=[],
         last_measured_best=[v for v in dep['summary'] if v['arm']=='A128D'])
(D/'sizing.json').write_text(json.dumps(out,indent=2)+'\n')
print('S19_MODELED_LOAD_US',ts,'FREE_WAIT_R',out['score_model']['free_wait_removal_R'],'COMMON_ADDED_COST_BUDGET',budget)
for v in options:print(v['kind'],v['added_output_path_proxy_us'],v['modeled_R_before_sync_underfill_finish'],v['sensitivity_R_if_proxy_halves_and_sync_is_half_us'])
print('DEPENDENCY_FORMS',len(table),'historical terms; lane status and new run are in ../ladder')
