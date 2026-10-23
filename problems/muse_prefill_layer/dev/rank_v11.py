"""Receipt-only H200 profile ranking against sim.bound's prefill model."""
from pathlib import Path
import sys,re,json,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from sim import bound
from sim.problems import spec,task_floors
P=Path(__file__).resolve().parents[1]
cases=spec('muse_prefill_layer')['benchmarks'];floors=task_floors('muse_prefill_layer')
jobs=[('check','6a9cd943fe404d74','cand_v11'),('scored','965a9a816d4744ea','promoted_6a9cd943fe404d74')]
def kind(name):
 if 'moe_gemm_kernel<1' in name:return 'expert gate/up'
 if 'moe_gemm_kernel<0' in name:return 'expert down'
 if 'deep_gemm::' in name:return 'expert BF16 grouped GEMM'
 if '_dequant' in name:return 'expert dequant'
 if '_swiglu' in name:return 'expert SwiGLU'
 if 'flash::' in name:return 'attention'
 if 'nvjet' in name:return 'dense/router GEMM'
 if '_boundary' in name:return 'residual boundary'
 return 'other visible'
result=[]
for mode,job,stem in jobs:
 text=(P/'profiles'/f'{stem}.{job}.txt').read_text()
 parts=re.split(r'\nprofile \d+:',text)[1:]
 loads={}
 for i,part in enumerate(parts):
  case=cases[i];T=case['batch']*case['q_len'];weight=case['weight'];load=case['load']
  kernels=[]
  for line in part.splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs\s+([\d.]+)%\s+(\d+)',line)
   if m:kernels.append(dict(name=m[1].strip(),us=float(m[2]),calls=int(m[4]),kind=kind(m[1])))
  total=float(re.search(r'\ntotal\s+([\d.]+) µs',part)[1])
  touched,routes=bound.inferred_routing('muse_prefill_layer',case,floors[i])
  ew=bound.nvfp4_expert_work(touched,routes,T,external_io=False)
  aw=bound.attention_work(case['batch'],case['q_len'],case['ctx'],case['layer'],external_io=False)
  dw=sum((bound.projection_work(dict(m=T,layer=n)) for n in bound.PROJ),bound.Work())
  work=bound.problem_shape_work('muse_prefill_layer',case,floors[i])
  model=dict(experts_us=ew.bf16_flops/bound.BF16*1e6,dense_us=dw.bf16_flops/bound.BF16*1e6,
             attention_us=(aw.bf16_flops/bound.BF16+aw.fp8_flops/bound.FP8)*1e6,
             tensor_us=bound.floor(work,bound.SPEC_BW).tensor_us,routes=routes,touched=touched)
  row=dict(case=case,total_us=total,kernels=kernels,floor=model)
  result.append(dict(mode=mode,job=job,**row))
  l=loads.setdefault(load,dict(total_us=0.,floor_us=0.,stage=collections.defaultdict(float),kernels=collections.defaultdict(float),count=0))
  l['total_us']+=weight*total;l['floor_us']+=weight*model['tensor_us'];l['count']+=weight
  for k in kernels:l['stage'][k['kind']]+=weight*k['us'];l['kernels'][k['name']]+=weight*k['us']
  l['stage']['unlisted']+=weight*(total-sum(k['us'] for k in kernels))
 print(mode,job)
 for name,l in loads.items():
  print(name,'weightedms',round(l['total_us']/1000,3),'floorms',round(l['floor_us']/1000,3),'ratio',round(l['total_us']/l['floor_us'],3))
  print(' stage',[(k,round(v/1000,3)) for k,v in sorted(l['stage'].items(),key=lambda a:-a[1])])
  print(' top kernels',[(k[:70],round(v/1000,3)) for k,v in sorted(l['kernels'].items(),key=lambda a:-a[1])[:5]])
(P/'dev/v11_profile_ranking.json').write_text(json.dumps(result,indent=2))
lines=['# V11 H200 profile ranking versus tensor floors','','Serves: G2, G1. Profile calls are single traces, not scored medians. All times below are judge evidence; floors are `sim.bound` model calculations (989 TFLOP/s BF16, 1979 TFLOP/s FP8 QK). Top-k profiles omit six small kernels per shape. The floor currently charges PV at BF16 peak; legal FP8 PV can lower its attention component.','', '| Run | Load/layer/prefix | Expert GEMM+dequant ms | Dense/router ms | Attention ms | Boundary ms | Total ms | Tensor floor ms | Expert / dense / attention floors ms |','|---|---|---:|---:|---:|---:|---:|---:|---|']
for r in result:
 st=collections.defaultdict(float)
 for k in r['kernels']:st[k['kind']]+=k['us']
 e=sum(v for k,v in st.items() if k.startswith('expert') and k!='expert SwiGLU')
 f=r['floor'];c=r['case']
 lines.append(f"| {r['mode']} | {c['load']}/{c['layer']}/{c['ctx']} | {e/1000:.3f} | {st['dense/router GEMM']/1000:.3f} | {st['attention']/1000:.3f} | {st['residual boundary']/1000:.3f} | {r['total_us']/1000:.3f} | {f['tensor_us']/1000:.3f} | {f['experts_us']/1000:.3f} / {f['dense_us']/1000:.3f} / {f['attention_us']/1000:.3f} |")
lines+=['','See `v11_profile_ranking.json` for each visible kernel, official shape weights and inferred routing. Check `6a9cd943fe404d74`; scored `965a9a816d4744ea`. Expert FLOP floors use the bound model\'s inferred mean routes, not observed counts from a profiled invocation.']
summary=['','## Weighted per-load ranking (ms)','','Profile totals exclude graph replay overhead; tensor floors here are replay-adjusted. The raw bound table adds 62/62/186 replays (0.262/0.262/0.786 ms).','','| Run | Load | Ranked visible stages (ms) | Total / tensor floor (ms) |','|---|---|---|---|']
for mode,job,_ in jobs:
 for load in ('turn 1','turn 4','first turn'):
  stage=collections.defaultdict(float);total=fl=0.
  for r in result:
   if r['mode']!=mode or r['case']['load']!=load:continue
   weight=r['case']['weight'];total+=weight*r['total_us'];fl+=weight*r['floor']['tensor_us']
   for k in r['kernels']:stage[k['kind']]+=weight*k['us']
  ranked=' > '.join(f'{k} {v/1000:.3f}' for k,v in sorted(stage.items(),key=lambda item:-item[1]) if k!='other visible')
  summary.append(f'| {mode} | {load} | {ranked} | {total/1000:.3f} / {fl/1000:.3f} |')
lines+=summary
(P/'dev/V11_PROFILE.md').write_text('\n'.join(lines)+'\n')
