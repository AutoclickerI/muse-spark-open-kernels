"""Render completed receipts and known pending jobs; never infers missing H200 times."""
import json,re,datetime,hashlib
from pathlib import Path
from collections import defaultdict
p=Path(__file__).resolve().parents[1];root=p.parents[1]
jobs={1:'300425fdfc3e4aed',2:'7ef49d84f4a14791',3:'82a4e50b54294a2a',4:'27f3e312727d423c',5:'84b5433a1f2f43e5',6:'1961a539127a4184',7:'a1252f4de2ef4a70',8:'a2702eaedce34844',9:'f13e757f4c224e06',10:'583a1b97cc544f53',11:'6a9cd943fe404d74',12:'53b79ddde4a048d5',13:'362f63065c634281'}
notes={1:'Inherited Marlin + dequant/DeepGEMM',2:'Norm/glue launch changes',3:'Custom WGMMA, 128-token tiles',4:'Contiguous FP8 cache gather',5:'Dense API trial (actually cuBLAS dispatch)',6:'Humming indexed experts',7:'Custom WGMMA, 64-token tiles',8:'Humming masked groups below2048',9:'Contiguous cache + unpacked GQA',10:'Route-indexed SwiGLU',11:'WGMMA below8192, dequant/DG above',12:'WGMMA 256-token tiles at large T',13:'Corrected native DeepGEMM dense'}
receipts={}
for f in (root/'runs/muse_prefill_layer').glob('*.json'):
 d=json.loads(f.read_text());receipts[d['id']]=(d,f)
def loads(r):
 sums=defaultdict(lambda:[0.,0.])
 for x in r.get('records',[]):
  if x.get('event')=='benchmark' and x.get('status')=='pass':
   c=x['case'];sums[c['load']][0]+=x['baseline_ns']*c['weight'];sums[c['load']][1]+=x['median_ns']*c['weight']
 return {k:a/b for k,(a,b) in sums.items()}
def values(r):
 ls=loads(r)
 return [f'{ls[k]:.3f}x' if k in ls else '—' for k in ('turn 1','turn 4','first turn')]+[f"{r['speedup']:.3f}x" if r.get('speedup') else '—']
now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
lines=['# Prefill experiments', '',f'Report generated {now}. H200 numbers below come only from judge receipts; local GPU times are excluded.', '', '| Candidate | Check job | Status | Turn 1 | Turn 4 | First turn | Overall |','|---|---|---|---:|---:|---:|---:|']
best=(0,None,None)
for n,jid in jobs.items():
 if jid not in receipts:
  lines.append(f'| v{n}: {notes[n]} | `{jid}` | pending | — | — | — | — |');continue
 d,f=receipts[jid];r=d.get('result',{});status='passed' if r.get('passed') else str(r.get('error') or r.get('status'))
 lines.append(f'| v{n}: {notes[n]} | [`{jid}`](../../{f.relative_to(root)}) | {status} | '+' | '.join(values(r))+' |')
 if r.get('passed') and not r.get('flags') and r.get('speedup',0)>best[0]:best=(r['speedup'],n,jid)
lines+=['','## Scored runs','','| Job | Status | Turn 1 | Turn 4 | First turn | Overall |','|---|---|---:|---:|---:|---:|']
for jid in ['facdb8724574420f','bab2f3468fcb4b58']:
 if jid in receipts:
  d,f=receipts[jid];r=d.get('result',{});lines.append(f'| [`{jid}`](../../{f.relative_to(root)}) | '+('passed' if r.get('passed') else str(r.get('error')))+' | '+' | '.join(values(r))+' |')
 else:lines.append(f'| `{jid}` | pending | — | — | — | — |')
if best[1]:
 d,f=receipts[best[2]];r=d['result'];src=f.with_suffix('.py')
 exact=(p/'submission.py').read_bytes()==src.read_bytes()
 lines+=['',f'Best completed check: **v{best[1]}, {best[0]:.6f}x**, 95% interval {r.get("speedup_ci")}. `submission.py` '+('matches its exact receipt source.' if exact else 'does not match that receipt; promotion/review remains pending.'),'','## Profile breakdown','', 'Single profiled calls, in microseconds; these differ from the paired benchmark medians. Expert core includes GEMMs and explicit dequantization; activation/routing/norm/gather/combine are in Other. Only kernels identifiable in the reported top-kernel table are assigned.','', '| Load / layer / prefix | Expert core | FA3 + cache gather | Dense + router GEMMs | Other | Total |','|---|---:|---:|---:|---:|---:|']
 for x in r.get('records',[]):
  if x.get('event')!='profile':continue
  st=defaultdict(float)
  for line in x['report'].splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs',line)
   if not m:continue
   name,us=m.group(1),float(m.group(2))
   if name=='total':st['total']=us
   elif 'flash::' in name or '_gather_cache' in name:st['attention']+=us
   elif any(s in name for s in ['Marlin','moe_gemm_kernel','_dequant_kernel','humming']):st['expert']+=us
   elif 'deep_gemm::' in name:st['expert' if best[1]!=13 else 'mixed']+=us
   elif name.startswith(('nvjet','ampere')):st['dense']+=us
  other=st['total']-st['expert']-st['attention']-st['dense']
  c=x['case'];label=f"{c['load']} / {c['layer']} / {c['ctx']}"
  lines.append('| '+label+' | '+' | '.join(f'{v:.1f}' for v in [st['expert'],st['attention'],st['dense'],other,st['total']])+' |')
 if best[1]==13:lines+=['','Native dense and grouped DeepGEMM names are truncated identically in these profiles; their aggregate is in Other. Do not infer an exact expert/dense split from this table.']
lines+=['','## Verification and porting','','All v1–v13 passed the five official local tests. Additional fresh-input CUDA graph tests cover nearby token counts, uneven lengths, page64 and KV scale1/64. Expert edge tests cover zero routes, one expert and eight experts. v12 was forced through its new256-token path on all five local tests and nearby graph tests. v13 was additionally forced through native dense kernels for local layer and graph tests. Hopper-only paths are validated only by their completed judge receipts.','', 'The first scored v1 entry was verified at rank46 (1.2124x) at10:08UTC. Re-query the board for the final rank; later scored jobs may still be pending.','', 'See [PORTING.md](PORTING.md) for exact SGLang functions, checkpoint scale/layout differences, collective boundaries, variable-length capture requirements and workspace considerations. No eight-H200 serving validation has run.','', 'New optimization branches stopped after the v2/v4 overall experiments failed to beat the best. Already-submitted runs are still being evaluated; v13 corrects the backend dispatch of the existing v5 experiment.']
(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(p/'RESULTS.md', 'best',best)
