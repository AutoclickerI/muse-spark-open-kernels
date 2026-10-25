"""Recover old receipts without relabeling their base or treating residence as wait cycles."""
import json,subprocess,statistics,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
paths={'trace':'problems/muse_bf16_proj/dev/producer_glue/h200_9845398be06b4c37.json','paired':'problems/muse_bf16_proj/dev/pdl_audit/h200_0f74021a2c53445e.json','sass':'problems/muse_bf16_proj/dev/pdl_audit/baseline_native.json'}
x={k:json.loads(subprocess.check_output(['git','show','ec55726:'+v])) for k,v in paths.items()}
trace=[]
for cell in x['trace']['results']:
 if cell['case'] not in ('d1S','d1F'):continue
 p=cell['profiles']['A'];st=p['stages'];roles={}
 for role,producer in [('b1','o'),('route','pool_router'),('b2','unpooler_slice')]:
  b=st[role];pred=st[producer];roles[role]=dict(residence_us=b['end']-b['start'],tail_after_producer_us=max(0.,b['end']-pred['end']),overlap_us=max(0.,min(b['end'],pred['end'])-b['start']),start_us=b['start'],end_us=b['end'],producer=producer,producer_end_us=pred['end'])
 trace.append(dict(case=cell['case'],roles=roles))
selected=[]
for row in x['sass']:
 if row['name']=='GLUE_CL_CUDA':
  for k in row['kernels']:
   if 'route_clILi8ELi256' in k['symbol'] or 'boundary_clILi8ELi256ELi1' in k['symbol']:selected.append(k)
paired=[]
for cell in x['paired']['cells']:
 if cell['case'] not in ('d1S','d1F'):continue
 vals={}
 for arm in ('A2','M','D'):
  v=[r[arm]/r['A'] for r in cell['samples']];vals[arm]=dict(latency_ratio=statistics.mean(v),ci95=1.96*statistics.stdev(v)/math.sqrt(len(v)))
 paired.append(dict(case=cell['case'],seeds=cell['seeds'],ratios=vals))
out=dict(timestamp_trace_job=x['trace']['job'],timestamp_trace_base=x['trace']['host_base'],timestamp_trace_is_s14=False,trace=trace,s14_paired=paired,shared_jit_control=False,s14_sass_selected=selected,s14_pair_job='0f74021a2c53445e',s14_pair_base=x['paired'].get('base_sha'),s14_timestamp_trace_available=False,warning='Existing S14 paired receipt has no per-role timestamps. S13 overlap is context only. New lab captures S14; no numerical wait cycles inferred.')
(HERE/'existing_receipts.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(trace=trace,warning=out['warning']),indent=2))
