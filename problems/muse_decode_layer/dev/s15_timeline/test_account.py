"""Meaningful overlap/join controls for the accounting, independent of kernel code."""
import json
from pathlib import Path
from analysis import chain_account
# Heavy and light run on independent streams; heavy determines the join.
trace=[dict(node=1,role='qkv_gate',name='gemm',stream=1,start_ns=0,end_ns=1000),dict(node=2,role='moe',name='mwbr::gemm<32, true>',stream=2,start_ns=1100,end_ns=6000),dict(node=3,role='moe',name='mwbr::gemm<16, true>',stream=3,start_ns=1100,end_ns=4000),dict(node=4,role='b2',name='norm',stream=1,start_ns=3000,end_ns=7000)]
dag=dict(nodes=[[1,0],[2,0],[3,0],[4,0]],edges=[[0,1,0,0,0],[0,2,0,0,0],[1,3,1,0,0],[2,3,1,0,0]])
a=chain_account(trace,dag);assert a['wall_us']==7 and a['critical_us']['qkv_gate']==1 and a['critical_us']['moe_heavy']==4.9 and a['critical_us'].get('moe_light',0)==0 and a['critical_us']['b2']==1 and abs(a['critical_us']['launch_gaps']-.1)<1e-9
# A release-only descendant can complete before its producer; no double charge.
trace[-1]['end_ns']=5000;a2=chain_account(trace,dag);assert a2['wall_us']==6 and a2['critical_us'].get('b2',0)==0
# Event/join nodes pass through the latest predecessor.
dag['nodes'].insert(3,[-4,5]);dag['edges']=[[0,1,0,0,0],[0,2,0,0,0],[1,3,0,0,0],[2,3,0,0,0],[3,4,0,0,0]];trace[-1].update(start_ns=6100,end_ns=7100);a3=chain_account(trace,dag);assert a3['wall_us']==7.1 and abs(sum(a3['critical_us'].values())-7.1)<1e-9
(Path(__file__).parent/'algorithm_validation.json').write_text(json.dumps(dict(fork_join=a,early_completion=a2,event_join=a3),indent=2)+'\n');print('PASS fork/join, PDL overlap, early completion and event join accounting')

# Device memset is work, not an uncovered launch interval.
mem=[dict(node=11,role='setup',name='cudaMemset',stream=1,start_ns=0,end_ns=500),dict(node=12,role='b1',name='norm',stream=1,start_ns=600,end_ns=1600)]
md=dict(nodes=[[11,2],[12,0]],edges=[[0,1,0,0,0]])
a4=chain_account(mem,md);assert a4['critical_us']['setup']==.5 and a4['critical_us']['b1']==1 and abs(a4['critical_us']['launch_gaps']-.1)<1e-9
print('PASS memset time charged to GPU work rather than launch gaps')

# A later-starting small branch can win completion while heavy work fills its
# logical dependency gap. The gap must not be reported as GPU idle.
from decode import gap_audit
fork=[dict(node=1,role='route',name='r',stream=1,start_ns=0,end_ns=1000),dict(node=2,role='moe_heavy',name='h',stream=2,start_ns=1000,end_ns=5000),dict(node=3,role='moe_small',name='s',stream=3,start_ns=4000,end_ns=7000)]
cell=dict(trace=fork,dag=dict(nodes=[[1,0],[2,0],[3,0]],edges=[[0,1,0,0,0],[0,2,0,0,0]]),trace_seed=1)
a5=gap_audit(cell);assert a5['dag_gaps_us']==3 and a5['gpu_idle_us']==0 and a5['busy_overlap_us']=={'moe_heavy':3}
fork[1]['start_ns']=2000;a6=gap_audit(cell);assert a6['gpu_idle_us']==1 and a6['busy_overlap_us']=={'moe_heavy':2}
print('PASS busy cross-stream overlap separated from genuine GPU idle')
