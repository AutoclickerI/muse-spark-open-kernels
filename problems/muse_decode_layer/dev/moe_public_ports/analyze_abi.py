"""Matched S19 ABI results and explicit integration candidates."""
import argparse,hashlib,json,statistics,sys
from pathlib import Path
from analyze import ci
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'));from lab_packet import paired_evidence
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();r=json.loads(a.receipt.read_text());d=paired_evidence(r['result']['log_tail'],'ABI_RESULT');manifest=json.loads((H/'abi_manifest.json').read_text());assert d['complete'] and not d['failures'];assert d['seeds']==manifest['seeds'] and d['identities']==manifest['identities']
whole={};phases={};aa={};selection={};mechanisms=[];fixed=d['fixed_ns']
for M,rows in d['round_ns'].items():
 means=[{k:statistics.mean(v)-fixed for k,v in row.items()} for row in rows];assert len(means)==8 and all(len(v)==4 for row in rows for v in row.values());aa[M]=ci([row['A2']/row['A'] for row in means]);whole[M]={}
 for n in rows[0]:
  if n in ('A','A2'):continue
  whole[M][n]=dict(**ci([row[n]/row['A'] for row in means]),mean_us=statistics.mean(row[n] for row in means)/1000)
 best=min(whole[M],key=lambda n:whole[M][n]['r']);selection[M]=best if whole[M][best]['r']<1 else 'A'
 pmeans=[{k:statistics.mean(v)-fixed for k,v in row.items()} for row in d['phase_ns'][M]];phases[M]={}
 for key in pmeans[0]:
  n,stage=key.split(':');den='A:'+stage;q=dict(mean_us=statistics.mean(row[key] for row in pmeans)/1000,base_us=statistics.mean(row[den] for row in pmeans)/1000)
  if n!='A':
   q.update(ci([row[key]/row[den] for row in pmeans]))
   if q['r']<1:mechanisms.append(dict(M=int(M),key=key,**q))
  phases[M][key]=q
 print('M',M,'whole',[(n,round(q['r'],6),[round(q['lo'],6),round(q['hi'],6)]) for n,q in whole[M].items()]);print('phases',[(n,round(q.get('r',1),6),[round(q.get('lo',1),6),round(q.get('hi',1),6)]) for n,q in phases[M].items()])
result=dict(measured=True,receipt=str(a.receipt.resolve().relative_to(R)),whole=whole,phases=phases,AA=aa,selection=selection,mechanisms=mechanisms,identities=d['identities'],seeds=d['seeds'],checks=d['checks'],elapsed_s=d['elapsed_s'],method='Eight fresh seed pairs, all4round arithmetic means; paired port/S19 t7 CI95. S19 ACT/Y/SS and norm/precision/metadata contracts matched.',promotion=False)
(H/'abi_receipt_data.json').write_text(json.dumps(d,indent=2)+'\n');(H/'abi_result.json').write_text(json.dumps(result,indent=2)+'\n')
fmt=lambda q:f"{q['r']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
lines=['# Matched S19 ABI follow-up','','The first whole-port cohort remains valid. Its native phase boundaries differ: O W2 omitted S19’s inline norm/SS, and Lee finalize read BF16 Y/scalar SS rather than FP32 Y/64 SS. This follow-up includes the actual S19 work and buffers. It does not treat the unmatched ratios as borrowable wins.','','| M | O W13 hybrid | O W2 hybrid | O W13+W2 | Lee finalizer hybrid | pick |','|---:|---:|---:|---:|---:|---|']
for M in ('128','256'):lines.append('| '+M+' | '+' | '.join(fmt(whole[M][n]) for n in ('O13','O2','O132','LF'))+' | '+selection[M]+' |')
lines+=['','| M | phase | port/S19 CI95 | port / S19 us |','|---:|---|---:|---:|']
for M in ('128','256'):
 for key,q in phases[M].items():
  if 'r' in q:lines.append(f"| {M} | {key} | {fmt(q)} | {q['mean_us']:.3f} / {q['base_us']:.3f} |")
lines+=['',f"Receipt `{result['receipt']}`; TASK checks `{d['checks']}`; {d['elapsed_s']:.1f}s.",'',result['method'],'']
(H/'ABI_H200.md').write_text('\n'.join(lines))
