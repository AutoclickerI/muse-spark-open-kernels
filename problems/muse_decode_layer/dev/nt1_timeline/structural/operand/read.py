"""Audit the immutable receipt and report conditional oracle contrasts; never infer G0 from a leaf."""
import argparse,ast,base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet,promotion
read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest();MS=(1,8,16,32)
CONTRASTS={
 'global_to_shared_fragment':('BASE_PAD','FRAG_SMEM'),
 'pack_free_register':('BASE','FRAG_REG'),
 'cached_register_lifetime':('FRAG_SMEM','FRAG_JIT'),
 'accumulator_recurrence':('ACC_CTRL','ACC_FREE'),
 'ideal_useful_MMA_count':('PAD_CTRL','PAD_USEFUL'),
 'MMA_activation':('L2','L1'),
 'epilogue_NOWAIT':('L3','L2')}
def stats(xs):
 n=len(xs);m=statistics.mean(xs);se=statistics.stdev(xs)/n**.5;e=promotion.T95[min(n-1,30)]*se
 return dict(mean=m,SE=se,CI95=[m-e,m+e],n=n,values=xs)

def analyze(outer,raw,*,h200):
 assert outer['complete'] and outer['seeds']==list(range(8)) and not raw['clock_error']
 assert raw['protocol']==dict(name='palindromic-cohort-v1',rounds=4,seed_count=8,aggregation='arithmetic-mean-all-rounds',selective_retiming=False)
 assert all(v['arms']==['LAZY'] and v['scope']=='arm' for v in outer['checks']['failures'])
 for arm in ('BASE','AA','BASE_PAD','ACC_CTRL','PAD_CTRL','HIST'):
  a,b=outer['checks']['gates'][arm];assert a<=.015 and b<=.075
 manifest=read(D/'payload.json');assert outer['identities']==manifest['identities']
 counts=read(D/'counts.json')['rows'];assert len(raw['resources'])==60
 for M,arm,v in raw['resources']:
  c=next(x for x in counts if x['M']==M and x['arm']==('BASE' if arm=='AA' else arm))
  if h200:assert v==[2,c['registers'],c['stack'],c['shared'],c['static'],264]
  else:assert v[3]==c['shared'] and v[0]>0 and v[5]>0
 sample={};rounds=[]
 for seed,M,delta,rems in raw['round_delta32_remainder_ns']:
  assert len(delta)==len(rems)==len(raw['arm_names'])*4
  ticks=0;values=[]
  for dd,rr in zip(delta,rems):ticks+=dd;values.append(ticks*32+rr)
  assert min(values)>0
  rr={arm:values[i*4:(i+1)*4] for i,arm in enumerate(raw['arm_names'])}
  sample[seed,M]={arm:statistics.mean(v)/1000 for arm,v in rr.items()};rounds.append(dict(seed=seed,M=M,ns=rr))
 assert set(sample)=={(s,m) for s in range(8) for m in MS}
 for key in ('telemetry','work_counts','histograms'):
  assert len(raw[key])==32 and {(v[0],v[1]) for v in raw[key]}==set(sample)
 for seed,M,t in raw['telemetry']:
  assert len(t)==3 and all(v['sm_MHz']>0 and v['mem_MHz']>0 for v in t)
  assert [v['t_s'] for v in t]==sorted(v['t_s'] for v in t)
 padding=[]
 for seed,M,nn in raw['work_counts']:
  assert len(nn)==32 and 0<sum(nn)<=M*8 and min(nn)>=0
  blocks=sum((v+7)//8 for v in nn);h=next(h for s,m,h in raw['histograms'] if (s,m)==(seed,M))
  if 'SM_columns' in h:h['SM_rows']=[list(v) for v in zip(*h.pop('SM_columns'))];h['CTA_joint']=[list(v) for v in zip(*h.pop('CTA_joint_columns'))]
  grid=264 if h200 else h['CTA_count'];assert h['CTA_count']==grid and sum(v[-1] for v in h['CTA_joint'])==grid
  tiles=sum(v[0]*v[-1] for v in h['CTA_joint']);assert tiles==768*blocks
  if seed==0:assert sum(v[1] for v in h['SM_rows'])==grid
  issued=tiles*8*32;useful_equivalent=768*32*sum(nn)
  padding.append(dict(seed=seed,M=M,pairs=sum(nn),blocks=blocks,tiles=tiles,issued_warp_HMMA=issued,useful_equivalent_warp_HMMA=useful_equivalent,useful_fraction=useful_equivalent/issued,scope='Source-derived dynamic count: executed CTA tile histogram times eight warps times32 retained HMMA sites;useful equivalent is not a legal narrower-MMA implementation'))
 summary=[];contrasts=[];ratios=[]
 for M in MS:
  for arm in raw['arm_names']:summary.append(dict(M=M,arm=arm,**stats([sample[s,M][arm] for s in range(8)])))
  for term,(old,new) in CONTRASTS.items():contrasts.append(dict(M=M,term=term,comparison=[old,new],**stats([sample[s,M][old]-sample[s,M][new] for s in range(8)])))
  aa=[sample[s,M]['AA']/sample[s,M]['BASE'] for s in range(8)]
  for arm in raw['arm_names']:
   rr=promotion.interval([sample[s,M][arm]/sample[s,M]['BASE'] for s in range(8)],aa)
   failed=arm=='LAZY' and any(v.get('M',M)==M for v in outer['checks']['failures'])
   ratios.append(dict(M=M,arm=arm,time_ratio=rr['R'],CI95=rr['CI95'],log_SE=rr['log_se'],AA_log_SE=rr['AA_log_se'],seeds=8,measured=h200 and not failed,diagnostic=arm not in ('BASE','AA','BASE_PAD','ACC_CTRL','PAD_CTRL','HIST','LAZY'),whole_production_graph_measured=False))
 t={int(k):v for k,v in read(D.parent/'ladder/results.json')['modeled_layer_us'].items()};ceilings=[]
 for term in CONTRASTS:
  deltas={M:next(v for v in contrasts if v['M']==M and v['term']==term) for M in MS}
  perM={M:math.exp(-math.log1p(-min(max(0,v['mean']),t[M]*.99)/t[M])/6) for M,v in deltas.items()}
  per_seed=[math.exp(sum(-math.log1p(-min(max(0,deltas[M]['values'][s]),t[M]*.99)/t[M]) for M in (8,16,32))/6) for s in range(8)]
  ceilings.append(dict(term=term,per_M_R=perM,M8_32_R=promotion.interval(per_seed),assumption='Planning only:full positive wall delta,S19 transported layer times;not held-out whole-load G0;contrast mechanisms overlap'))
 return dict(measured=h200,scope='native H200 diagnostic' if h200 else 'sm120 screen only;H200 timings unmeasured',summary=summary,contrasts=contrasts,ratios=ratios,padding=padding,ceilings=ceilings,rounds=rounds,raw=raw,checks=outer['checks'],P4_rows=[],next='Use largest buildable oracle term to select a correct arm,then pair it with frozen S21 in production graphs with all six loads and fresh seeds under promotion.evaluate;no leaf-only nomination')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('receipt',nargs='?',type=Path);ap.add_argument('--local-screen',action='store_true');a=ap.parse_args()
 if a.local_screen:
  assert a.receipt is None
  d=read(R/'runs/screen/p3a_operand/local_packet.json');out=analyze(d['outer'],d['decoded'],h200=False);(R/'runs/screen/p3a_operand/local_analysis.json').write_text(json.dumps(out,indent=2)+'\n');print('Local packet analysis passed;no H200 timings emitted');return
 assert a.receipt is not None
 host=read(a.receipt);payload=read(D/'payload.json');q=read(D/'queue.json');assert host['result']['gpu']=='H200' and host['mode']=='check' and host['board']==q['host']
 source=a.receipt.resolve().with_suffix('.py');text=source.read_text();assert sha(text.encode())==q['wrapper_sha256']
 if q.get('job'):assert host['id']==q['job']
 entries=next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets));name,short,body=next(v for v in entries if v[1]==payload['sha256'][:16]);assert sha(body.encode())==payload['sha256']
 blob=next(n.args[0].value for n in ast.walk(ast.parse(body)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');expanded=lzma.decompress(base64.b85decode(blob));assert sha(expanded)==payload['expanded_sha256']
 for c in payload['cells'].values():
  s=(R/c['cell']).read_bytes();assert sha(s)==c['cell_sha256'] and s in expanded
 rec=next(v for v in lab_packet.read_packets(host['result']['log_tail']) if v['source_sha256']==short);outer=rec['packet'];assert outer['kind']=='P3A_OPERAND' and not outer.get('transport_errors');raw=json.loads(lzma.decompress(base64.b85decode(outer['paired']['data'])))
 out=analyze(outer,raw,h200=True);out.update(job=host['id'],source_audit=dict(verified=True,wrapper_sha256=q['wrapper_sha256'],payload_sha256=payload['sha256'],packet_checksum=rec['checksum']),host_passed=host['result'].get('passed'))
 (D/'results.json').write_text(json.dumps(out,indent=2)+'\n');q.update(job=host['id'],state='complete;source/control/resources/rounds/telemetry/work verified',receipt=str(a.receipt.resolve().relative_to(R)));(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
 print('H200_ORACLES',host['id'])
 for M in MS:print('M',M,[(v['term'],round(v['mean'],3),[round(x,3) for x in v['CI95']]) for v in out['contrasts'] if v['M']==M])
if __name__=='__main__':main()
