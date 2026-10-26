"""One cohort:retain every round,verify sources/controls,and report paired ladder contrasts."""
import argparse,ast,base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet,promotion
read=lambda p:json.loads(p.read_text())
def stats(xs):
 n=len(xs);m=statistics.mean(xs);se=statistics.stdev(xs)/n**.5;err=promotion.T95[min(n-1,30)]*se
 return dict(mean=m,SE=se,CI95=[m-err,m+err],n=n,values=xs)
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();host=read(a.receipt);q=read(D/'queue.json');payload=read(D/'payloads.json');counts=read(D/'counts.json')['rows']+read(D/'early/counts.json')['rows'];assert host['id']==q['job']
source=(R/q['source']).read_text();assert hashlib.sha256(source.encode()).hexdigest()==q['sha256'];entries=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets));spec=payload['payloads'][0];name,sha,body=next(v for v in entries if v[1]==spec['sha256'][:16]);assert hashlib.sha256(body.encode()).hexdigest()==spec['sha256']
blob=next(n.args[0].value for n in ast.walk(ast.parse(body)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');expanded=lzma.decompress(base64.b85decode(blob)).decode();assert hashlib.sha256(expanded.encode()).hexdigest()==spec['expanded_sha256']
for c in payload['identities']['cells'].values():
 text=(R/c['cell']).read_text();assert hashlib.sha256(text.encode()).hexdigest()==c['cell_sha256'] and text in expanded
records=lab_packet.read_packets(host['result']['log_tail']);rec=next(v for v in records if v['source_sha256']==sha);out=rec['packet'];assert out['kind']=='P3A_LADDER' and out['paired']['codec']=='lzma-json-b85' and not out.get('transport_errors');raw=json.loads(lzma.decompress(base64.b85decode(out['paired']['data'])));raw.update(complete=out['complete'],checks=out['checks'],seeds=out['seeds']);assert raw['complete'] and raw['seeds']==list(range(8)) ;assert all(v.get('arms')==['EARLY'] for v in raw['checks']['failures']);assert raw['identities_sha256']==hashlib.sha256(json.dumps(payload['identities'],sort_keys=True).encode()).hexdigest();protocol=raw['protocol'];assert protocol==dict(name='palindromic-cohort-v1',rounds=4,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
spec1=payload['payloads'][1];entry1=next(v for v in entries if v[1]==spec1['sha256'][:16]);assert hashlib.sha256(entry1[2].encode()).hexdigest()==spec1['sha256'];histrec=next(v for v in records if v['source_sha256']==spec1['sha256'][:16]);hr=histrec['packet'];assert hr['kind']=='P3A_LADDER_HIST' and hr['complete'] and hr['seeds']==[0] and not hr.get('transport_errors') and not hr['checks']['failures'];assert hr['identities']==out['identities']
histpaired=json.loads(lzma.decompress(base64.b85decode(hr['paired']['data'])));hr['paired']=histpaired
for _,M,h in hr['paired']['histograms']:
 h['SM_rows']=[list(v) for v in zip(*h.pop('SM_columns'))];h['CTA_joint']=[list(v) for v in zip(*h.pop('CTA_joint_columns'))];assert h['CTA_count']==264 and sum(v[1] for v in h['SM_rows'])==264
raw['seed0_SM_diagnostic']=hr
for arm in ('BASE','AA','BSINK','HIST'):
 assert raw['checks']['gates'][arm][0]<=.015 and raw['checks']['gates'][arm][1]<=.075
assert len(raw['resources'])==60
for M,arm,info in raw['resources']:
 c=next(v for v in counts if v['M']==M and v['arm']==('BASE' if arm=='AA' else arm));assert info==[2,c['registers'],c['stack'],c['shared'],c['static'],264],(M,arm,info,c)
samples={};rounds={}
for seed,M,delta,rems in raw['round_delta32_remainder_ns']:
 n=len(raw['arm_names'])*4;assert len(delta)==n+1 and len(rems)==n;last=delta[0];values=[]
 for d,r in zip(delta[1:],rems):last+=d;values.append(last*32+r)
 assert all(v>0 for v in values)
 rr={arm:values[i*4:(i+1)*4] for i,arm in enumerate(raw['arm_names'])};rounds[(seed,M)]=rr;samples[(seed,M)]={arm:statistics.mean(v)/1000 for arm,v in rr.items()}
assert len(samples)==32 and set(samples)=={(s,m) for s in range(8) for m in (1,8,16,32)}
assert len(raw['histograms'])==len(raw['telemetry'])==len(raw['work_counts'])==32
assert raw['clock_error'] is None and len(raw['host_enqueue_top4'])==4
for key in ('histograms','telemetry','work_counts'):
 assert {(v[0],v[1]) for v in raw[key]}==set(samples)
for seed,M,snapshots in raw['telemetry']:
 assert len(snapshots)==3 and all(set(v)=={'t_s','sm_MHz','mem_MHz','temperature_C','power_mW'} for v in snapshots)
 assert all(v['sm_MHz']>0 and v['mem_MHz']>0 for v in snapshots)
 assert [v['t_s'] for v in snapshots]==sorted(v['t_s'] for v in snapshots)
for seed,M,counts32 in raw['work_counts']:
 assert len(counts32)==32 and 0<sum(counts32)<=M*8 and all(v>=0 for v in counts32)
assert len(hr['paired']['histograms'])==4
for seed,M,h in raw['histograms']:
 assert h['CTA_count']==264 and sum(v[-1] for v in h['CTA_joint'])==264
 if 'SM_rows' in h:assert sum(v[1] for v in h['SM_rows'])==264
summary=[];contrasts=[];tails=[]
names={'decode_scale':('L1','L0S'),'MMA_activation':('L2','L1'),'epilogue_NOWAIT':('L3','L2'),'dependency':('BSINK','L3'),'live_sink_normal':('BSINK','BASE'),'live_sink_raw':('L0S','L0'),'load_issue_repeats':('L0','STREAM'),'activation_transport_sink':('ACT','LOOP'),'tile_schedule':('LOOP','META')}
for M in (1,8,16,32):
 for arm in raw['arm_names']:
  v=stats([samples[(s,M)][arm] for s in range(8)]);summary.append(dict(M=M,arm=arm,**v))
 aa=promotion.interval([samples[(s,M)]['AA']/samples[(s,M)]['BASE'] for s in range(8)])
 for term,(new,old) in names.items():
  v=stats([samples[(s,M)][new]-samples[(s,M)][old] for s in range(8)]);contrasts.append(dict(M=M,term=term,comparison=[new,old],**v))
 h=[h for seed,m,h in raw['histograms'] if m==M];tails.append(dict(M=M,observer=promotion.interval([samples[(s,M)]['HIST']/samples[(s,M)]['BASE'] for s in range(8)]),AA=aa,W2_tail_p50_to_last_us=stats([(v['endpoints_ns'][4]-v['endpoints_ns'][3])/1000 for v in h]),finish_tail_p50_to_last_us=stats([(v['endpoints_ns'][7]-v['endpoints_ns'][6])/1000 for v in h]),entry_spread_us=stats([(v['endpoints_ns'][1]-v['endpoints_ns'][0])/1000 for v in h])))
 # Telescoping identity is checked per seed,not by mixing cohorts or ratios of means.
 for s in range(8):
  t=samples[(s,M)];assert abs(t['BASE']-(t['L0S']+(t['L1']-t['L0S'])+(t['L2']-t['L1'])+(t['L3']-t['L2'])+(t['BSINK']-t['L3'])-(t['BSINK']-t['BASE'])))<1e-7
s18=read(R/'runs/muse_decode_layer/20261026T050310Z_submit_28b5c189e7b5469c.json');s19=read(R/'problems/muse_decode_layer/dev/s19/batches/22f25954e6ee84ab/report.json');loads={1:'decode 1',8:'verify 1',16:'decode 16',32:'decode 32'};latencies={};ceilings=[]
for M,load in loads.items():
 rr=[v for v in s18['result']['records'] if v.get('event')=='benchmark' and v['case']['load']==load];t=sum(v['case']['weight']*v['median_ns']/1000 for v in rr)/62/statistics.mean(v['INC'] for v in s19['raw']['loads'][load]['samples']);latencies[M]=t
 for v in contrasts:
  if v['M']!=M:continue
  d=max(0,v['mean']);ceilings.append(dict(M=M,load=load,term=v['term'],stage_delta_us=d,modeled_R=math.exp(-math.log1p(-min(d,t*.99)/t)/6)))
candidate=[]
for M in (1,8,16,32):
 failed=any(v.get('M')==M or v.get('M') in (9,24) for v in raw['checks']['failures'])
 rr=[samples[(seed,M)]['EARLY']/samples[(seed,M)]['BASE'] for seed in range(8)];aa=[samples[(seed,M)]['AA']/samples[(seed,M)]['BASE'] for seed in range(8)];ci=promotion.interval(rr,aa)
 candidate.append(dict(M=M,arm='EARLY',ratio=ci['R'],CI95=ci['CI95'],log_SE=ci['log_se'],AA_log_SE=ci['AA_log_se'],seeds=8,seed_indices=list(range(8)),seed_ratios=rr,control_alias=M==1,measured=not failed,point_nominee=not failed and M!=1 and ci['R']<1,promotion='Stage point nomination only;P4 confirms held-out whole-layer G0 R'))
result=dict(job=q['job'],measured=True,invalid=[],host_passed=host['result'].get('passed'),source_audit=dict(verified=True,wrapper_sha256=q['sha256'],payload_sha256=spec['sha256'],expanded_sha256=spec['expanded_sha256'],packet_checksum=rec['checksum'],histogram_packet_checksum=histrec['checksum']),protocol=protocol,candidate=candidate,summary=summary,contrasts=contrasts,tail= tails,modeled_layer_us=latencies,ceilings=ceilings,rounds=[dict(seed=s,M=m,ns=rr) for (s,m),rr in sorted(rounds.items())],seed_level_record=raw,notes='All rounds retained;diagnostic cuts not candidates;sink/activation observer costs explicit;ceilings full wall credit,not G0 proof')
(D/'results.json').write_text(json.dumps(result,indent=2)+'\n');q.update(state='complete;source/control/resources/all-rounds/telemetry/histograms verified',receipt=str(a.receipt.resolve().relative_to(R)));(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
for M in (1,8,16,32):
 print('M',M,' '.join(v['arm']+'='+str(round(v['mean'],3)) for v in summary if v['M']==M))
 for v in contrasts:
  if v['M']==M:print(v['term'],round(v['mean'],3),[round(x,3) for x in v['CI95']])

for v in candidate:print('CANDIDATE',v['M'],v['ratio'],v['CI95'],v['point_nominee'],v['measured'])
