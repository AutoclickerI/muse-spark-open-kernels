"""Render completed receipts and known pending jobs; never infers missing H200 times."""
import ast,json,re,datetime,hashlib
from pathlib import Path
from collections import defaultdict
p=Path(__file__).resolve().parents[1];root=p.parents[1]
jobs={1:'300425fdfc3e4aed',2:'7ef49d84f4a14791',3:'82a4e50b54294a2a',4:'27f3e312727d423c',5:'84b5433a1f2f43e5',6:'1961a539127a4184',7:'a1252f4de2ef4a70',8:'a2702eaedce34844',9:'f13e757f4c224e06',10:'583a1b97cc544f53',11:'6a9cd943fe404d74',12:'53b79ddde4a048d5',13:'362f63065c634281'}
notes={1:'Inherited Marlin + dequant/DeepGEMM',2:'Norm/glue launch changes',3:'Custom WGMMA, 128-token tiles',4:'Contiguous FP8 cache gather',5:'Dense API trial (actually cuBLAS dispatch)',6:'Humming indexed experts',7:'Custom WGMMA, 64-token tiles',8:'Humming masked groups below2048',9:'Contiguous cache + unpacked GQA',10:'Route-indexed SwiGLU',11:'WGMMA below8192, dequant/DG above',12:'WGMMA 256-token tiles at large T',13:'Corrected native DeepGEMM dense'}
notes.update({14:'TMA indexed Humming experts',15:'Shared/shared WGMMA expert pipeline',16:'Grouped GEMM expected-row tuning',17:'Lossless bit-plane WGMMA decoding',18:'Measured stage integration'})
# New jobs are assigned only after admission. Recover their IDs from durable rows.
for f in [root/'runs/ledger.jsonl', *sorted((root/'runs/pending').glob('*.json'))]:
 for line in (f.read_text().splitlines() if f.suffix=='.jsonl' else [f.read_text()]):
  try: row=json.loads(line)
  except json.JSONDecodeError: continue
  if row.get('problem')!='muse_prefill_layer' or row.get('mode')!='check': continue
  m=re.search(r'cand_v(\d+)\.py$',row.get('file',''))
  if m: jobs[int(m.group(1))]=row['job']
for n in range(14,18):jobs.setdefault(n,None)
if (p/'cand_v18.py').exists():jobs.setdefault(18,None)
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
for n,jid in sorted(jobs.items()):
 if jid not in receipts:
  job=f'`{jid}`' if jid else 'not assigned'
  status='pending' if jid else 'waiting admission'
  if n==18 and jid is None and (p/'dev/finish_round_state.json').exists():
   state=json.loads((p/'dev/finish_round_state.json').read_text())
   status=(state.get('integration') or {}).get('status',status)
  lines.append(f'| v{n}: {notes.get(n,"candidate")} | {job} | {status} | — | — | — | — |');continue
 d,f=receipts[jid];r=d.get('result',{});status='passed' if r.get('passed') else str(r.get('error') or r.get('status'))
 lines.append(f'| v{n}: {notes[n]} | [`{jid}`](../../{f.relative_to(root)}) | {status} | '+' | '.join(values(r))+' |')
 if r.get('passed') and not r.get('flags') and r.get('speedup',0)>best[0]:best=(r['speedup'],n,jid)
lines+=['','## Scored runs','','| Job | Status | Turn 1 | Turn 4 | First turn | Overall |','|---|---|---:|---:|---:|---:|']
score_jobs=['facdb8724574420f','bab2f3468fcb4b58']
for jid,(d,f) in receipts.items():
 if d.get('mode')=='submit' and jid not in score_jobs:score_jobs.append(jid)
for f in (root/'runs/pending').glob('*.json'):
 row=json.loads(f.read_text())
 if row.get('problem')=='muse_prefill_layer' and row.get('mode')=='submit' and row['job'] not in score_jobs:score_jobs.append(row['job'])
for jid in score_jobs:
 if jid in receipts:
  d,f=receipts[jid];r=d.get('result',{});lines.append(f'| [`{jid}`](../../{f.relative_to(root)}) | '+('passed' if r.get('passed') else str(r.get('error')))+' | '+' | '.join(values(r))+' |')
 else:lines.append(f'| `{jid}` | pending | — | — | — | — |')
if best[1]:
 d,f=receipts[best[2]];r=d['result'];src=f.with_suffix('.py')
 modes=re.search(r'^_DENSE_MODES = (.+)$',src.read_text(),re.M)
 native_dense=best[1]==13 or (modes is not None and 'deepgemm' in ast.literal_eval(modes.group(1)).values())
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
   elif 'deep_gemm::' in name:st['mixed' if native_dense else 'expert']+=us
   elif name.startswith(('nvjet','ampere')):st['dense']+=us
  other=st['total']-st['expert']-st['attention']-st['dense']
  c=x['case'];label=f"{c['load']} / {c['layer']} / {c['ctx']}"
  lines.append('| '+label+' | '+' | '.join(f'{v:.1f}' for v in [st['expert'],st['attention'],st['dense'],other,st['total']])+' |')
 if native_dense:lines+=['','Native dense and grouped DeepGEMM names are truncated identically in these profiles; their aggregate is in Other. Do not infer an exact expert/dense split from this table.']
lines+=['','## Verification and porting','','All v1–v17 passed the five official local tests. Additional fresh-input CUDA graph tests cover nearby token counts, uneven lengths, page64 and KV scale1/64. Expert edge tests cover zero routes, one expert and eight experts. v12 was forced through its new256-token path on all five local tests and nearby graph tests. v13 was additionally forced through native dense kernels for local layer and graph tests. v15 shared-A gate arithmetic is exercised by mma.sync emulation. v16 was also forced through its DG branch on all tests. v17 passed an actual GEMM probe spanning all finite signed E4M3 scales and all E2M1 codes (0.159% relative L2 locally). v15/v17 sm90a builds have no spills. Hopper-only paths are validated only by their completed judge receipts.','', 'See [PORTING.md](PORTING.md) for exact SGLang functions, checkpoint scale/layout differences, collective boundaries, variable-length capture requirements and workspace considerations. No eight-H200 serving validation has run.','', 'The coordinator revoked the two-failed-ideas stopping condition. Background collectors recover pending receipts. dev/finish_round.py promotes and scores only a checked new best from this round; the final check is reserved for measured component integration after the component receipts arrive. Four new checks v14–v17 reserve usage to19/20 including waiting admission; accepted usage may be lower.']
board=p/'dev/board_resumed.json'
if board.exists():
 b=json.loads(board.read_text());you=b.get('you',{})
 stamp=datetime.datetime.fromtimestamp(board.stat().st_mtime,datetime.timezone.utc).isoformat(timespec='seconds')
 lines+=['',f'Board verified {stamp}: rank {you.get("rank")}, {you.get("speedup")}x. Later scored jobs can still be pending.']
lines+=['','## Next ideas, ranked','','1. Combine the fast bit-plane decoder with256-token tiles if the separate H200 results support both. Keep one common packed bank.','2. Dequantize the selected packed bank directly into grouped-GEMM scratch so a custom/DG serving hybrid does not retain another model-wide NVFP4 layout.','3. Revisit BN64 with lower register limits and fewer pipeline stages to test two-CTA occupancy. v7 changed tile size while retaining the large register allocation; its measured regression does not establish the best attainable BN64 configuration.','','These are hypotheses, not measured H200 improvements. Pending receipts determine their priority.']
(p/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(p/'RESULTS.md', 'best',best)
