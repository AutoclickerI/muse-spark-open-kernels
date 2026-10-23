"""Build STATUS and per-stage H200 tables from receipts only; no API calls."""
import collections,json,re
from pathlib import Path
root=Path(__file__).resolve().parents[3]
dev=Path(__file__).parent
loads=['decode 1','decode 16','decode 32','verify 1','verify 16','verify 32']
rows=[];profiles=[]
for line in (root/'runs/ledger.jsonl').read_text().splitlines():
 row=json.loads(line)
 if row['problem']!='muse_decode_layer':continue
 src=root/row['source'];f=src.with_suffix('.json')
 if not f.exists():continue
 d=json.loads(f.read_text());result=d.get('result') or {}
 acc=collections.defaultdict(lambda:[0.,0.])
 for r in result.get('records',[]):
  if r['event']=='benchmark' and r.get('median_ns'):
   c=r['case'];a=acc[c['load']];wt=c.get('weight',1)
   a[0]+=wt*r['baseline_ns'];a[1]+=wt*r['median_ns']
  if r['event']=='profile': profiles.append((row,r))
 ratios=[f'{acc[l][0]/acc[l][1]:.3f}' if acc[l][1] else '—' for l in loads]
 status='pass' if result.get('passed') else ('duplicate' if d.get('status')=='duplicate' else 'FAIL')
 errors=[r.get('message','') for r in result.get('records',[]) if r.get('status')=='fail']
 if errors:status+=': '+errors[0].replace('|','/')[:85]
 overall=result.get('speedup');overall=f'{overall:.4f}' if overall else '—'
 rows.append((row['utc'],f"| {Path(row['file']).name} ({row['mode']}) | [{row['job']}](../../../{f.relative_to(root)}) | {status} | {overall} | "+' | '.join(ratios)+' |'))
seen={json.loads(l)['job'] for l in (root/'runs/ledger.jsonl').read_text().splitlines() if json.loads(l)['problem']=='muse_decode_layer'}
pending=[]
for f in (root/'runs/pending').glob('*.json'):
 r=json.loads(f.read_text())
 if r['problem']=='muse_decode_layer':pending.append(r);seen.add(r['job'])
known={'69a655d3b2364fc4':'cand_adapt2.py','a5ab4a2011f44321':'cand_adapt3.py','b2a8999311034eb3':'cand_adapt4.py'}
for j,n in known.items():
 if j not in seen:pending.append({'job':j,'file':n,'mode':'check','utc':''})
for r in pending:
 rows.append((r['utc'],f"| {Path(r['file']).name} ({r['mode']}) | {r['job']} | pending, no H200 result | — | — | — | — | — | — | — |"))
text='''# Decode layer status, 2026-10-23

Goal: improve the measured whole decoder layer and submit any check that beats the best
verified decode entry. All ratios below are H200 receipt measurements; `—` is unmeasured.
The new candidates retain NVFP4, BF16 activation boundaries, FP8 KV, all attention keys,
and the permitted BF16 hi+lo / FP32 router. No commits were made.

## H200 per-load results

Load ratios aggregate sliding/full latencies with official weights before division.

| Candidate | Job / receipt | Result | Overall | D1 | D16 | D32 | V1 | V16 | V32 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
'''
text+='\n'.join(r for _,r in sorted(rows,key=lambda z:z[0]))+'\n\n'
statepath=dev/'promotion_state.json'
text+='## Live promotion state\n\n'
if statepath.exists():
 state=json.loads(statepath.read_text())
 text+=f"Automatic scored promotions launched: {len(state['promotions'])}.\n\n"
 for entry in state['promotions']:
  text+=f"- Check {entry['check_job']}, {entry['candidate']}, verified {entry['speedup']:.4f}x; [submission log](../../../{entry['log']}).\n"
 if state.get('failures'):text+='\nPromotion errors: '+str(state['failures'])+'\n'
else:text+='No automatic scored promotion has been launched.\n'
text+='\n'
notes=dev/'STATUS_NOTES.md'
if notes.exists():text+=notes.read_text()
(dev/'STATUS.md').write_text(text)
out=['# H200 stage times from judge profiles','', 'Times are profiler GPU kernel sums in µs, not isolated stage benchmarks. Groups can overlap', 'under PDL, and truncated judge reports leave some kernels in the remainder.','', '| Candidate / job | Load / layer | Projections | Attention incl. prep/combine | Experts incl. activation/combine | Boundaries | Route/norm | Profile total |','|---|---|---:|---:|---:|---:|---:|---:|']
for row,r in profiles:
 sums=collections.defaultdict(float)
 for line in r['report'].splitlines():
  m=re.search(r'\s+([\d.]+) µs',line)
  if not m:continue
  val=float(m[1]);name=line[:m.start()].lower()
  if name.strip()=='total':group='total'
  elif any(x in name for x in ['gemv','skinny','nvjet','splitkreduce']):group='proj'
  elif any(x in name for x in ['attn_main','attn_combine','qk_prep','flash','norm_gate']):group='attn'
  elif any(x in name for x in ['marlin','moe_fused','silu','act_and_mul','post_expert']):group='moe'
  elif 'boundary' in name:group='boundary'
  elif 'route_norm' in name:group='route'
  else:group='other'
  sums[group]+=val
 c=r['case'];out.append(f"| {Path(row['file']).name} / {row['job']} | {c['load']} / {c['layer']} | "+' | '.join(f'{sums[k]:.1f}' for k in ['proj','attn','moe','boundary','route','total'])+' |')
(dev/'H200_STAGES.md').write_text('\n'.join(out)+'\n')
print(dev/'STATUS.md')
