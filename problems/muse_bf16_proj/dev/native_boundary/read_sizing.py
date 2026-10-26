"""Size two ceilings on native S19 timestamps, with explicit overlap limits."""
import base64,json,lzma,math,statistics,sys,struct
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
exec((H/'codec.py.inc').read_text())
P=Path(sys.argv[1])
if not P.exists():P=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(P.read_text());transport=read_packets(d['result']['log_tail']);assert transport and all(not x['packet'].get('transport_errors') and x['packet'].get('complete') is not False for x in transport),'Incomplete or failed wrapper';lines=result_lines(d['result']['log_tail']);blob=next(x.split(' ',1)[1] for x in lines if x.startswith('NATIVE_SIZE_XZ85 '));raw=nb_decode(blob)
assert raw['complete'],raw['failure'];rows=[];cells={}
fit=json.loads((H.parent/'stream_gap/decomposition.json').read_text())['stream_fit'];bw=fit['marginal_TBs'];intercept=fit['intercept_us']
def ci(v,log=False):
 z=[math.log(x) for x in v] if log else v;mu=statistics.mean(z);se=statistics.stdev(z)/math.sqrt(len(z));z=[mu,mu-2.364624251*se,mu+2.364624251*se];return [math.exp(x) for x in z] if log else z
for cell in raw['rows']:
 key=tuple(cell['case']);t=key[0]*key[1];per=[]
 for si,pair in enumerate(cell['profiles']):
  mean=lambda getter:statistics.mean(getter(p) for p in pair)/1000
  wall=mean(lambda p:p['wall_ns']);pe=ce=cp=service=0
  for role,meta in cell['weights'].items():
   res=mean(lambda p:p['residence_ns'].get(role,0));crit=mean(lambda p:p['critical_ns'].get(role,0));roof=intercept+meta['bytes']/1e6/bw;ex=max(res-roof,0);pe+=ex;cp+=min(ex,crit);service+=max(crit-roof,0)
   if si==0:rows.append(dict(case=key,t=t,role=role,weight_bytes=meta['bytes'],encoding=meta['encoding'],splits=meta['splits'],P_us=[],critical_us=[],roof_us=roof,excess_us=[]))
   r=next(x for x in rows if tuple(x['case'])==key and x['role']==role);r['P_us'].append(res);r['critical_us'].append(crit);r['excess_us'].append(ex)
  cons=sum(mean(lambda p,role=role:p['residence_ns'].get(role,0)) for role in ('b1','route','b2'));conscrit=sum(mean(lambda p,role=role:p['critical_ns'].get(role,0)) for role in ('b1','route','b2'))
  per.append(dict(wall=wall,producer_excess=pe,producer_exposed_cap=cp,producer_critical_excess=service,consumer_all=cons,consumer_critical_cap=conscrit))
 cells[key]=per
for r in rows:
 for k in ('P_us','critical_us','excess_us'):r[k+'_CI']=ci(r[k])
terms={}
for label,small in (('small_M',True),('large_M_issue_split',False)):
 seedR=[];loads=[]
 for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
  samples=[];removed=[];is_selected=(b*q<=32)==small
  for i in range(8):
   sides=[cells[b,q,w][i] for w in (2048,0)];old=sum(z['wall']*count for z,count in zip(sides,(46,16)));delta=sum((z['producer_critical_excess']+(0 if small else z['consumer_critical_cap']))*count for z,count in zip(sides,(46,16))) if is_selected else 0
   assert old>delta;removed.append(delta/62);samples.append(old/(old-delta))
  loads.append(dict(load=[b,q],removed_us_per_layer=ci(removed),R=ci(samples,True),samples=samples))
 for i in range(8):seedR.append(math.exp(statistics.mean(math.log(x['samples'][i]) for x in loads)))
 terms[label]=dict(R=ci(seedR,True),loads=loads,meaning='completion-critical producer contribution above weight-only fitted stream; scenario, no additional prefetch overlap assumed' if small else 'completion-critical producer excess plus complete b1/route/b2 removal; upper scenario, split-only consumer portion unmeasured')
out=dict(job=d['id'],raw=raw,fit=fit,rows=rows,terms=terms,selected=max(terms,key=lambda k:terms[k]['R'][0]),limits=['CUPTI native kernel timestamps add no device work; profiled vs unprofiled event controls retained','Ceilings are sizing models, not measured counterfactual speedups or G0; no CI can validate a counterfactual assumption','Raw residence includes PDL wait overlapping predecessors (O16F103.8us residence vs6.08us critical), so residence-minus-roof is not an independent saving','Large consumer full-deletion bound is deliberately optimistic; split-only portion measured next','Physical HBM bytes/counters, exclusive native issue waits and physical activation/tensor/epilogue split unmeasured'])
(H/'sizing_results.json').write_text(json.dumps(out,indent=2)+'\n')
for k,v in terms.items():print(k,v['R'],[(x['load'],x['removed_us_per_layer'][0]) for x in v['loads']])
print('SELECTED',out['selected'])
