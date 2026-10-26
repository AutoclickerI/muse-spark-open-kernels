"""Reproducible compulsory-byte, launch and achieved-rate decomposition."""
import json,re,statistics,math,csv,datetime
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
files=list((R/'runs/muse_moe_w4a16').glob('*2015fc90abf94e14.json'));assert len(files)==1
raw=json.loads(files[0].read_text());result=raw['result'];tail=result['log_tail'];body=re.search(r'^LAB_BEGIN payload .*?\n(.*?)^LAB_END',tail,re.M|re.S)[1]
assert 'END complete=30/30' in body and not any(s in body for s in ('LAB_ERROR','TRUNCATED','INCOMPLETE'))
streams=[]
for q in re.finditer(r'^STREAM (\d+) ([\d.]+) \[([\d.]+),([\d.]+)\]',body,re.M):
 b,t,lo,hi=map(float,q.groups());streams.append(dict(bytes=int(b),us=t,min_us=lo,max_us=hi,effective_TBs=b/t/1e6))
large=[x for x in streams if x['bytes']>=32<<20];mx=statistics.mean(x['bytes']/1e6 for x in large);my=statistics.mean(x['us'] for x in large);slope=sum((x['bytes']/1e6-mx)*(x['us']-my) for x in large)/sum((x['bytes']/1e6-mx)**2 for x in large)
fit=dict(intercept_us=my-slope*mx,marginal_TBs=1/slope,max_effective_TBs=max(x['effective_TBs'] for x in streams),published_best_stream_TBs=4.58,hardware_peak_TBs=4.8)
nop=[list(map(float,q.split(':'))) for q in re.search(r'^NOP (.*)',body,re.M)[1].split(',')];ns=[p for p in nop if p[0]>=2];nx=statistics.mean(p[0] for p in ns);ny=statistics.mean(p[1] for p in ns);cost=sum((n-nx)*(v-ny) for n,v in ns)/sum((n-nx)**2 for n,v in ns)
rows=[]
for match in re.finditer(r'^R (\d+):(\S+) A=([\d.]+) D=([\d.]+) F=([\d.]+) P=([\d.]+) dr=([\d.,]+) fr=([\d.]+) aa=([\d.,]+) nodes=([\d,]+) bytes=(\d+),(\d+) splits=(\d+) e=([\deE+.-]+)',body,re.M):
 t,role,a,d,f,p,dr,fr,aa,nodes,ab,db,s,e=match.groups();t=int(t);a,d,f,p,fr,e=map(float,(a,d,f,p,fr,e));ab,db,s=map(int,(ab,db,s));n,k={'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),'unpooler':(1024 if t<=8 else 8192,4096),'router':(256,8192)}[role];nodes=list(map(int,nodes.split(',')));dr=list(map(float,dr.split(',')));aa=list(map(float,aa.split(',')))
 changed=t<=32 or role=='router';nominate=changed and dr[0]<1
 row=dict(t=t,role=role,n=n,k=k,A_us=a,D_us=d,F_us=f,P_us=p,ratio=dr,AA_ratio=aa,nodes_A=nodes[0],nodes_D=nodes[2],nodes_P=nodes[-1],A_weight_bytes=ab,D_weight_bytes=db,A_bytes_per_weight=ab/(n*k),D_bytes_per_weight=db/(n*k),A_bits_per_weight=8*ab/(n*k),D_bits_per_weight=8*db/(n*k),A_effective_TBs=ab/a/1e6,D_effective_TBs=db/d/1e6,A_fraction_of_best_stream=ab/a/1e6/4.58,A_fraction_of_peak=ab/a/1e6/4.8,producer_effective_TBs=ab/p/1e6,producer_fraction_of_best_stream=ab/p/1e6/4.58,producer_fraction_of_peak=ab/p/1e6/4.8,producer_residual_vs_stream_fit_us=p-fit['intercept_us']-ab/1e6/fit['marginal_TBs'],weight_bytes_saved=ab-db,splits=s,A_tail_us=a-p,generic_cast_cost_us=f-d if f else None,flop_floor_us=2*t*n*k/989e6,nomination=nominate,layer_applicability='standalone diagnostic; layer router is fused into pool_router' if role=='router' else 'pool_router has a separate fused FP32 handoff; this N4096 board result is not a new layer nomination' if role=='pooler' else 'board conversion mostly absent from FP32 layer consumers; require neighbour chain pair',caveat='T1 timed one-hot only; generic kernel work is value independent, random confirmation required' if t==1 else 'Eight independent input seeds at one synthetic weight seed')
 rows.append(row)
assert len(rows)==30
out=dict(job=raw['id'],receipt=str(files[0].relative_to(R)),valid=True,outer_host_passed=result['passed'],updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stream_fit=fit,nop=dict(raw=nop,marginal_launch_us=cost,first_launch_minus_empty_us=nop[1][1]-nop[0][1]),rows=rows,limits=['weight bytes are complete compulsory banks, not hardware DRAM counters','vendor tiled issue multiplicity inferred from profile names, not DRAM traffic','leader per-shape records and implementation are private; exact causal attribution to that team unavailable','generic direct-store candidate changes geometry vs A; F/D isolates its cast boundary','same-code unchanged large cells are calibration controls, not gains'])
# This diagnostic point selection is not a G0 or a held-out composition CI.
loads=[]
for t in (1,8,16,32,128,256):
 rr=[r for r in rows if r['t']==t];a=sum(r['A_us'] for r in rr);d=sum(min(r['A_us'],r['D_us']) if r['nomination'] else r['A_us'] for r in rr);loads.append(dict(t=t,A_sum_us=a,point_selected_sum_us=d,point_speedup=a/d))
out['component_point_selection']=dict(loads=loads,speedup=math.exp(statistics.mean(math.log(x['point_speedup']) for x in loads)),CI=None,caveat='optimistic same-receipt selection; joint samples were not retained in the first lab; no composition promotion')
(H/'decomposition.json').write_text(json.dumps(out,indent=2)+'\n')
with (H/'decomposition.csv').open('w',newline='') as f:
 fields=['t','role','A_us','D_us','P_us','ratio','AA_ratio','A_weight_bytes','D_weight_bytes','A_bytes_per_weight','D_bytes_per_weight','A_effective_TBs','D_effective_TBs','producer_effective_TBs','producer_fraction_of_best_stream','producer_fraction_of_peak','producer_residual_vs_stream_fit_us','A_fraction_of_best_stream','A_fraction_of_peak','nodes_A','nodes_D','nodes_P','splits','A_tail_us','generic_cast_cost_us','flop_floor_us'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
print('stream',fit,'launch us',cost,'point component selection',out['component_point_selection']['speedup'])
print('changed stage/board nominations',[(r['t'],r['role'],r['ratio']) for r in rows if r['nomination']])
