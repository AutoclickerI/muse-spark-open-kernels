"""Reproducible process audit and serving sensitivity using local evidence only."""
import collections, datetime, itertools, json, math, re, statistics, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sim import serving as s, muse, hw
from sim.calibrate import sglang_mean
from sim.problems import geomean
sys.path.insert(0,str(ROOT/'tools'));import profile_report as pr
D={p:pr.build(p) for p in pr.PROBLEMS}
(ROOT/'prof/report.json').write_text(json.dumps(list(D.values()),indent=2)+'\n')
ledger={j['job']:j for l in (ROOT/'runs/ledger.jsonl').read_text().splitlines() if (j:=json.loads(l))}
allj={}
for p in ROOT.glob('runs/muse_*/*.json'):
 j=json.loads(p.read_text());allj[j['id']]={'path':str(p.relative_to(ROOT)),**j}
process=[]
for p in pr.PROBLEMS:
 js=[j for j in allj.values() if j.get('board')==p and j.get('duplicate_of') is None and j.get('started_at') and j.get('finished_at')]
 qs=[(j['started_at']-j['created_at'])/60 for j in js];rs=[(j['finished_at']-j['started_at'])/60 for j in js]
 latest=max((j for j in js if j.get('limits')),key=lambda j:j['created_at'])
 process.append(dict(problem=p,completed=len(js),queue_min=min(qs),queue_median=statistics.median(qs),queue_max=max(qs),run_median=statistics.median(rs),run_max=max(rs),latest_limits=latest.get('limits'),latest_limit_job=latest['id'],jobs=[dict(id=j['id'],queue_min=(j['started_at']-j['created_at'])/60,runtime_min=(j['finished_at']-j['started_at'])/60) for j in js]))
# Pair exact source hashes where both local and remote source snapshots exist.
local=[]
for p in ROOT.glob('problems/*/dev/local*.json'):
 try: j=json.loads(p.read_text())
 except (ValueError,OSError):continue
 if isinstance(j,dict):
  local.append(dict(path=str(p.relative_to(ROOT)),keys=list(j),data=j))
(ROOT/'prof/local_schema.json').write_text(json.dumps(local,indent=2)+'\n')
# Component partitions: calibration profile shares allocate unprofiled paired judge medians.
# These are modeled allocations, not measured exclusive elapsed time.
def stock_parts(problem):
 d=D[problem]['sglang'];means={pr.case_key(r['case']):r['sglang_us'] for r in sglang_mean(problem)}
 rows=[]
 for v in d['loads'].values():
  for sh in v['shapes']:
   b=sh['case'];t=means[pr.case_key(b)];comp=collections.defaultdict(float)
   for r in sh['profile_rows']:comp[pr.group(r['name'])]+=r['us']/sh['profile_us']*t
   rows.append((b,dict(comp)))
 return rows
layer_rows=stock_parts('muse_decode_layer');prefill_rows=stock_parts('muse_prefill_layer')
def parts_at(c):
 parts=collections.defaultdict(float)
 for layer,n in [('sliding',46),('full',16)]:
  rows=[(b,v) for b,v in layer_rows if b.get('q_len')==8 and b['layer']==layer]
  total=s._loglog({b['batch']:sum(v.values()) for b,v in rows},c)
  gs=set().union(*(set(v) for b,v in rows))
  raw={g:s._loglog({b['batch']:max(1e-9,v.get(g,0)) for b,v in rows},c) for g in gs}
  for g,t in raw.items():parts[g]+=n*total*t/sum(raw.values())
 extra=s.layer_stack_us(c,'stock')-sum(parts.values());parts['moe']+=extra
 parts.update(s.extras_us('stock',c,0))
 return dict(parts)
args=SimpleNamespace(tau=4.16,turns=13,host_us=0.0,interactivity='e2e')
stock=s.curve('stock',args)
prefill={}
for load in ('turn 1','turn 4','first turn'):
 parts=collections.defaultdict(float)
 for b,v in prefill_rows:
  if b['load']==load:
   for g,t in v.items():parts[g]+=b['weight']*t
 prefill[load]=dict(parts)
# Ratios from paired H200 records on best receipts. Per-layer or per-component port effects
# extrapolate isolated problem ratios into serving and are NOT serving measurements.
def ratio(problem,load,layer=None):
 rows=[x for x in D[problem]['ours']['loads'][load]['shapes'] if layer is None or x['case']['layer']==layer]
 # our shapes retain median; paired baseline reconstructed directly from matching receipt.
 receipt=json.loads((ROOT/D[problem]['ours']['receipt']).read_text());rs={pr.case_key(r['case']):r for r in receipt['result']['records'] if r.get('event')=='benchmark'}
 return sum(rs[pr.case_key(x['case'])]['baseline_ns']*x['case']['weight'] for x in rows)/sum(rs[pr.case_key(x['case'])]['median_ns']*x['case']['weight'] for x in rows)
def layer_gain_port(problem,c):
 return s._loglog({k:ratio(problem,'verify '+str(k)) for k in (1,16,32)},c)
def bf16_extra(c):
 touched,_=s.layer_routing(c*8)
 return 2*s._gemm_us('stock',touched*(s.BF16_EXPERT_BYTES-muse.EXPERT_BYTES),0)
def portable_layer_work(c,component=None):
 if component=='moe':
  return (parts_at(c)['moe']-bf16_extra(c))*60/62
 if component:
  return parts_at(c)[component]
 # Conservative port policy: keep both BF16 layers fully stock.
 return (s.layer_stack_us(c,'stock')-bf16_extra(c))*60/62
def shifted_curve(savings,psave=0):
 pref=s.prefill_load_us('first turn','stock')+12*s.prefill_load_us('turn 1','stock')-psave
 out={};nt=13*220
 for c in s.RUNGS:
  step=stock[c]['step_us']-savings[c];window=c*pref+nt/args.tau*step
  out[c]=dict(step_us=step,per_gpu=nt*c/window*1e6/8,per_user=nt/window*1e6,prefill_share=c*pref/window)
 return out
sensitivity={}
for label,p,comp in [('MoE best v6 isolated ratio → layer experts','muse_moe_w4a16','moe'),('attention v2 isolated ratio → attention','muse_attention_decode','attention'),('proj sweep1 isolated ratio → dense','muse_bf16_proj','dense'),('decode layer v3 whole layer','muse_decode_layer',None)]:
 saves={c:portable_layer_work(c,comp)*(1-1/layer_gain_port(p,c)) for c in s.RUNGS}
 cv=shifted_curve(saves);sensitivity[label]=dict(savings_us=saves,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])

old=D['muse_moe_w4a16']['ours']
v4path=next((ROOT/'runs/muse_moe_w4a16').glob('*84c97d9483b34573.json'))
D['muse_moe_w4a16']['ours']=pr.one_run('muse_moe_w4a16',v4path,json.loads(v4path.read_text()),False)
saves={c:portable_layer_work(c,'moe')*(1-1/layer_gain_port('muse_moe_w4a16',c)) for c in s.RUNGS}
cv=shifted_curve(saves);sensitivity['MoE v4 isolated ratio → layer experts']=dict(savings_us=saves,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
D['muse_moe_w4a16']['ours']=old

# Shape-level attention hybrid: keep stock wherever candidate loses (every verify row here).
sensitivity['attention v2 verify-gated (keep stock)']=dict(savings_us={c:0 for c in s.RUNGS},score=1.0,capped=1.0)
# Prefill v3 whole load ratios against stock, retaining the official workload weights.
psave=60/62*sum(mult*s.prefill_load_us(load,'stock')*(1-1/ratio('muse_prefill_layer',load)) for load,mult in [('first turn',1),('turn 1',12)])
cv=shifted_curve({c:0 for c in s.RUNGS},psave)
sensitivity['prefill v3 whole layer']=dict(prefill_saved_us=psave,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
saves=sensitivity['decode layer v3 whole layer']['savings_us'];cv=shifted_curve(saves,psave)
sensitivity['decode v3 + prefill v3 (disjoint phases)']=dict(savings_us=saves,prefill_saved_us=psave,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
# Current planned serving changes mentioned in SERVING.md, sensitivity independent of kernel ports.
for label,ms in [('finalize low-token 0.7ms',0.7),('finalize low-token 1.5ms',1.5),('drafter fc sharding 0.15ms',0.15),('CPU +100us',-0.1),('CPU +1ms',-1.0)]:
 saves={c:ms*1000*(1 if 'low-token' not in label or c==1 else 0) for c in s.RUNGS};cv=shifted_curve(saves)
 sensitivity[label]=dict(savings_us=saves,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
# Row-split proposal ~0.5s less first-turn prefill (SERVING.md).
cv=shifted_curve({c:0 for c in s.RUNGS},500000)
sensitivity['prefill row-split assumed 0.5s']=dict(prefill_saved_us=500000,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
# Pooler token gate at <=128 tokens = c<=16.
saves={c:s.layer_stack_us(c,'practical')-s.layer_stack_us(c,'practical',True) if c<=16 else 0 for c in s.RUNGS};cv=shifted_curve(saves)
sensitivity['pooler/unpooler shard <=128 tokens (practical rate deltas on stock)']=dict(savings_us=saves,score=s.score(cv,stock)[0],capped=s.score(cv,stock,True)[0])
result=dict(process=process,serving_components={c:parts_at(c) for c in s.RUNGS},prefill_components=prefill,port_sensitivity=sensitivity,stock_curve=stock)
(ROOT/'prof/analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
