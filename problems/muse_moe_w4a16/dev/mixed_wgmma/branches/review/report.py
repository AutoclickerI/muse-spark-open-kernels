"""Decode frozen reset receipts and compare to the predictions made before sending.

Matches frozen payload hashes, including h200_lab's automatic merged wrappers.
Never infers a job from recency.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import re

H=Path(__file__).resolve().parent;B=H.parent;ROOT=B.parents[4]
BATCHES={
 'stage3':('f867a8b74a243d1e','cluster_schedule_payload','c5840d07ddb95a2d'),
 'round2':('71a7126383e39952','payload','67eca44eb93d9531'),
 'round3':('77150f5f5046c333','tail_payload','abc1631d5d23bdcb'),
 'round4':('8c4c89c1d51b179d','ws_payload','80c5d1c8fd50dc5a'),
 'round5':('f0c92b1867e9aa50','tokpair_payload','b610039eebfd98d7'),
}
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|nan|inf'

def decode(tail,payload):
 blocks=re.findall(r'^LAB_BEGIN (\S+)[^\n]*\n(.*?)(?:^LAB_END([^\n]*)|\Z)',tail,re.M|re.S)
 selected=[(text,end) for name,text,end in blocks if name==payload]
 if not selected:return dict(valid=False,reason='No payload markers in receipt',tail=tail[-2000:])
 text,end=selected[-1];rows=[]
 for line in text.splitlines():
  match=re.match(r'^M(128|256) (\S+) n=(\d+) ',line)
  if not match:continue
  values={k:float(v) for k,v in re.findall(r'(us|pairS14|pairCTA2|pair)=('+NUMBER+r')',line,re.I)}
  errors=re.search(r'L2/row=('+NUMBER+r')/('+NUMBER+r')',line,re.I)
  spread=re.search(r'lo/hi=('+NUMBER+r')/('+NUMBER+r')',line,re.I)
  gate='TASK=PASS' in line and errors is not None and float(errors[1])<=.015 and float(errors[2])<=.075
  rows.append(dict(m=int(match[1]),arm=match[2],n=int(match[3]),**values,L2=float(errors[1]) if errors else None,row=float(errors[2]) if errors else None,spread=list(map(float,spread.groups())) if spread else None,gate=gate,line=line))
 aa=re.search(r'AA signed/abs/max=('+NUMBER+r')/('+NUMBER+r')/('+NUMBER+r')',text,re.I)
 finish=re.search(r'END seeds=(\d+) used=('+NUMBER+r') left=('+NUMBER+r')',text,re.I)
 errors=[line for line in text.splitlines() if 'LAB_ERROR' in line or 'TASK=FAIL' in line]
 expected=8 if payload in ('ws_payload','tokpair_payload') else 10
 valid=len(rows)==expected and all(r['gate'] for r in rows) and aa is not None and finish is not None and not errors
 return dict(valid=valid,reason='complete paired lab, all gates pass' if valid else 'incomplete or failed lab/control; no performance closure',rows=rows,AA_us=dict(zip(('signed_mean','absolute_mean','max_absolute'),map(float,aa.groups()))) if aa else None,seeds=int(finish[1]) if finish else None,lab_seconds=float(finish[2]) if finish else None,errors=errors,lines=text.splitlines(),wrapper_end=end)

def predictions(batch,m):
 key=str(m)
 if batch=='stage3':
  # These are calibration observations, not topology-arm forecasts.
  return {},dict(prior_direct_vs_S14={128:1.1970,256:1.1322}[m])
 if batch=='round5':return {},dict(meaning='Decode-sharing diagnostic; no numeric latency forecast before native measurement. See exact traffic model in round5.')
 q=json.loads((B/batch/'model.json').read_text())[key]
 if batch=='round2':return {name:a['estimated_whole_same_overlap_us'] for name,a in q['arms'].items()},dict(S14_us=q['observed']['control'])
 if batch=='round3':return {name:a['point_us'] for name,a in q['arms'].items()},dict(S14_us=q['arms']['S14']['point_us'])
 return dict(CTA2=q['CTA2_point_us'],WS_PIPE=q['WS_PIPE_prediction_us'],WS_TILE_BARRIER=q['WS_TILE_BARRIER_prediction_us']),dict(S14_us={128:185.43,256:261.03}[m],optimistic_PIPE_us=q['optimistic_spatial_overlap_prediction_us'])

def main():
 def reading(path):
  try:return json.loads(path.read_text())
  except FileNotFoundError:return None  # queue/receipt transfer is concurrent
  except json.JSONDecodeError:
   print('INCOMPLETE_JSON awaiting completed write:',path)
   return None
 index=reading(H/'receipt_index.json') if (H/'receipt_index.json').exists() else {}
 index=index or {};records=list(index.values())
 try:ledger=(ROOT/'runs/ledger.jsonl').read_text()
 except FileNotFoundError:
  print('TRANSIENT_MISSING ledger: awaiting repository write/rename');ledger=''
 for number,line in enumerate(ledger.splitlines(keepends=True),1):
  if not line.endswith('\n'):
   print('INCOMPLETE_JSON awaiting ledger newline:',number);continue
  records.append(json.loads(line))
 records+=[q for p in (ROOT/'runs/pending').glob('*.json') if (q:=reading(p)) is not None]
 waiting=[q for p in (ROOT/'runs/waiting').glob('*.json') if (q:=reading(p)) is not None]
 def embedded(record):
  sha=record.get('sha256','');file=record.get('file','')
  snap=ROOT/'runs/sent'/sha[:16]/Path(file).name
  if not snap.exists() and record.get('source'):snap=ROOT/record['source']
  if not snap.exists():return []
  values=[]
  for node in ast.parse(snap.read_text()).body:
   if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_LAB_PAYLOADS' for t in node.targets):values=ast.literal_eval(node.value)
  for name,short,src in values:assert hashlib.sha256(src.encode()).hexdigest().startswith(short)
  return [(name,short) for name,short,src in values]
 candidates=[r for r in records+waiting if 'P3b' in r.get('note','')]
 embeds={r['sha256']:embedded(r) for r in candidates}
 out={}
 for batch,(sender,payload,expected) in BATCHES.items():
  matches=[r for r in records if (payload,expected) in embeds.get(r.get('sha256'),[])]
  if not matches:
   pending=[r for r in waiting if (payload,expected) in embeds.get(r.get('sha256'),[])]
   out[batch]=dict(state='waiting' if pending else 'missing_sender',sender=pending[-1]['sha256'][:16] if pending else sender,legacy_sender=sender,payload_sha256_prefix=expected,queue=pending)
   continue
  record=matches[-1];path=(ROOT/record['source']).with_suffix('.json');base=dict(sender=record['sha256'][:16],legacy_sender=sender,payload_sha256_prefix=expected,job=record['job'],receipt=str(path.relative_to(ROOT)))
  index[batch]=record
  if not path.exists():out[batch]=dict(base,state='at_judge');continue
  receipt=reading(path)
  if receipt is None:out[batch]=dict(base,state='at_judge',reason='Awaiting complete receipt write');continue
  result=receipt['result'];decoded=decode(result.get('log_tail',''),payload)
  base.update(state='measured' if decoded['valid'] else 'unmeasured',host_status=result.get('status'),decoded=decoded)
  comparisons=[]
  if decoded['valid']:
   for m in (128,256):
    rr=[r for r in decoded['rows'] if r['m']==m];control=next(r for r in rr if r['arm'] in ('S14','S14_NF'))
    forecasts,calibration=predictions(batch,m)
    for r in rr:
     pred=forecasts.get(r['arm']);ratio=r.get('pairS14',r.get('pair'))
     # Absolute drift and recipe-model error are separate. No inferred causal
     # split into decode/sync/epilogue or critical-path overlap is manufactured.
     c=dict(m=m,arm=r['arm'],observed_us=r['us'],observed_ratio=ratio,predicted_us=pred,calibration=calibration)
     if pred is not None and 'S14_us' in calibration:
      c.update(predicted_ratio=pred/calibration['S14_us'],control_normalized_prediction_us=pred*control['us']/calibration['S14_us'],observed_over_normalized_prediction=r['us']/(pred*control['us']/calibration['S14_us']))
     comparisons.append(c)
  base['comparisons']=comparisons;out[batch]=base
 for name,value in (('results.json',out),('receipt_index.json',index)):
  temp=H/(name+'.'+str(os.getpid())+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(H/name)
 for batch,q in out.items():
  print(batch,q['state'],q.get('job',q['sender']))
  if 'decoded' in q:print(' AA',q['decoded'].get('AA_us'),'seeds',q['decoded'].get('seeds'),'valid',q['decoded']['valid'])
  for c in q.get('comparisons',[]):print(' ',c['m'],c['arm'],'us',c['observed_us'],'ratio',c['observed_ratio'],'prediction',c['predicted_us'],'normalized_error_factor',c.get('observed_over_normalized_prediction'))
 return out

if __name__=='__main__':main()
