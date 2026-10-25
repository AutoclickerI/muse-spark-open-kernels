"""Observed completion-critical chain, with PDL overlap clipped rather than charged twice."""
import math,statistics
STREAM_BPS=4.58e12
STREAM_INTERCEPT_US=2.36
BF16_FLOPS=989e12
FP8_FLOPS=1979e12
HBM_BPS=4.8e12

def stage_name(role,name):
    if role!='moe':return role
    if 'mwbr::gemm<16,' in name:return 'moe_light'
    if 'mwbr::gemm<32,' in name:return 'moe_heavy'
    if 'pmz::phase<' in name or 'moe_var<' in name or 'moe_fused<' in name:return 'moe_small'
    if 'route' in name:return 'moe_meta'
    if 'finish' in name:return 'moe_combine'
    raise AssertionError(('unclassified expert kernel',name))

def chain_account(trace,dag):
    """Longest observed completion chain through the DAG, including implicit sink.

    Kernel end can precede a programmatic predecessor's end: its contribution
    then is zero. Gaps only count between the selected predecessor frontier and
    the next kernel's start. Nonkernel join/event nodes pass predecessor time.
    """
    assert trace and all(n['end_ns']>=n['start_ns']>=0 for n in trace)
    by={n['node']:n for n in trace};nodes=dag['nodes'];edges=dag['edges'];pred=[[] for _ in nodes];succ=[[] for _ in nodes]
    for a,b,*_ in edges:pred[b].append(a);succ[a].append(b)
    indegree=[len(p) for p in pred];queue=[i for i,d in enumerate(indegree) if not d];order=[]
    for i in queue:
        order.append(i)
        for j in succ[i]:
            indegree[j]-=1
            if not indegree[j]:queue.append(j)
    assert len(order)==len(nodes),'capture DAG contains a cycle'
    assert {n[0] for n in nodes if n[1] in (0,1,2)}==set(by),'every captured GPU kernel/copy/memset must have an activity record'
    origin=min(n['start_ns'] for n in trace);wall=max(n['end_ns'] for n in trace)-origin
    frontier={};parent={};contrib={};gaps={}
    for i in order:
        p=max(pred[i],key=lambda j:frontier[j],default=None);prior=frontier[p] if p is not None else origin;parent[i]=p
        n=by.get(nodes[i][0]);end=n['end_ns'] if n else prior
        frontier[i]=max(prior,end)
        gap=max(0,n['start_ns']-prior) if n else 0;gaps[i]=gap
        contrib[i]=max(0,end-max(prior,n['start_ns'])) if n else 0
    sink=max(order,key=lambda i:frontier[i]);path=[]
    while sink is not None:path.append(sink);sink=parent[sink]
    path.reverse();critical={};residence={};path_nodes=[]
    for i in path:
        n=by.get(nodes[i][0])
        if not n:continue
        stage=stage_name(n['role'],n['name']);critical[stage]=critical.get(stage,0)+contrib[i]/1000
        path_nodes.append(dict(node=n['node'],stage=stage,critical_us=contrib[i]/1000,gap_us=gaps[i]/1000))
    critical['launch_gaps']=sum(gaps[i] for i in path)/1000
    for n in trace:
        stage=stage_name(n['role'],n['name']);residence[stage]=residence.get(stage,0)+(n['end_ns']-n['start_ns'])/1000
    joins=[]
    for i in path:
        if len(pred[i])>1:
            ranked=sorted(((frontier[j],j) for j in pred[i]),reverse=True)
            joins.append(dict(node=nodes[i][0],winner=nodes[ranked[0][1]][0],runner_up=nodes[ranked[1][1]][0],slack_us=(ranked[0][0]-ranked[1][0])/1000))
    accounted=sum(critical.values());wall_us=wall/1000
    assert abs(accounted-wall_us)<=max(.0001,.02*wall_us),(accounted,wall_us)
    # Launch start is monotone along each same-stream dependency edge. PDL may
    # overlap end times, so end-before-next-start is deliberately not required.
    for a,b,*_ in edges:
        x,y=by.get(nodes[a][0]),by.get(nodes[b][0])
        if x and y and x['stream']==y['stream']:assert y['start_ns']>=x['start_ns'],(x,y)
    return dict(joins=joins,wall_us=wall_us,critical_us=critical,residence_us=residence,path=path_nodes,accounting_error_fraction=abs(accounted-wall_us)/max(wall_us,1e-9),monotone=True)

def model_roof(work,stream=True):
    reads=work.get('read_bytes',0);writes=work.get('write_bytes',0)
    compute=(work.get('bf16_flops',0)/BF16_FLOPS+work.get('fp8_flops',0)/FP8_FLOPS)*1e6
    byte_floor=(reads+writes)/HBM_BPS*1e6
    memory=(STREAM_INTERCEPT_US+reads/STREAM_BPS*1e6) if stream and reads else byte_floor
    return dict(byte_floor_us=byte_floor,flop_floor_us=compute,roof_us=max(memory,compute,byte_floor))

def shape_summary(cell):
    seeds=[]
    for seed in cell['seeds']:
        account=seed['account'];rows={}
        for stage in sorted(set(account['critical_us'])|set(seed['work'])):
            critical=account['critical_us'].get(stage,0.);w=seed['work'].get(stage,{})
            roof=model_roof(w) if stage!='launch_gaps' else dict(roof_us=0.,byte_floor_us=0.,flop_floor_us=0.)
            rows[stage]=dict(critical_us=critical,**w,**roof,above_roof_us=max(0.,critical-roof['roof_us']))
        seeds.append(dict(seed=seed['seed'],wall_us=account['wall_us'],rows=rows))
    stages=sorted({k for s in seeds for k in s['rows']});means={}
    for stage in stages:
        means[stage]={k:statistics.mean(s['rows'].get(stage,{}).get(k,0.) for s in seeds) for k in ('critical_us','read_bytes','write_bytes','bf16_flops','fp8_flops','byte_floor_us','flop_floor_us','roof_us','above_roof_us')}
    wall=statistics.mean(s['wall_us'] for s in seeds)
    return dict(case=cell['case'],weight=cell['weight'],wall_us=wall,rows=means,seeds=seeds)
