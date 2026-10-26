"""Receipt -> full timeline, per-stage/load critical work and PLAN Where-the-time-goes table."""
import argparse,base64,csv,json,lzma,statistics,struct
from pathlib import Path
from analysis import shape_summary,chain_account,model_roof
LOADS=('d1','d16','d32','v1','v16','v32')
GROUPS={'attention':('attention_sliding','attention_full'),'MoE':('moe_small','moe_heavy','moe_light','moe_meta','moe_combine'),'glue':('b1','b2','route','setup'),'projections':('qkv_gate','qkv3','gate','o','pool_router','unpooler','unpooler_slice'),'DAG gaps':('launch_gaps',)}

def gap_audit(cell):
 """Partition logical-path gaps by actual GPU activity; overlap is not causation."""
 trace=cell['trace'];account=chain_account(trace,cell['dag']);by={n['node']:n for n in trace};covered={};idle=0.
 for step in account['path']:
  end=by[step['node']]['start_ns'];start=end-round(step['gap_us']*1000)
  if start==end:continue
  cuts=sorted({start,end}|{max(start,min(end,n[k])) for n in trace for k in ('start_ns','end_ns')})
  for a,b in zip(cuts,cuts[1:]):
   active=tuple(sorted({n['role'] for n in trace if n['start_ns']<b and n['end_ns']>a}))
   if active:covered[active]=covered.get(active,0.)+(b-a)/1000
   else:idle+=(b-a)/1000
 assert abs(idle+sum(covered.values())-account['critical_us']['launch_gaps'])<1e-7
 return dict(seed=cell['trace_seed'],wall_us=account['wall_us'],dag_gaps_us=account['critical_us']['launch_gaps'],gpu_idle_us=idle,busy_overlap_us={'+'.join(k):v for k,v in covered.items()},critical_us=account['critical_us'])

def representative_owners(cases):
 """Owner totals on retained median-seed traces, with busy gaps assigned only to their owner.

 The activity intervals identify an owner, not which concurrent branch caused
 a scheduling delay. Never extrapolate this split onto the unretained seeds.
 """
 owner={s:g for g,stages in GROUPS.items() if g!='DAG gaps' for s in stages};shapes=[]
 for c in cases:
  audit=gap_audit(c);costs={g:sum(audit['critical_us'].get(s,0.) for s in stages) for g,stages in GROUPS.items() if g!='DAG gaps'}
  for active,value in audit['busy_overlap_us'].items():
   groups={owner[s] for s in active.split('+')};group=next(iter(groups)) if len(groups)==1 else 'unattributed busy overlap'
   costs[group]=costs.get(group,0.)+value
  costs['GPU idle']=audit['gpu_idle_us'];work=next(s['work'] for s in c['seeds'] if s['seed']==c['trace_seed']);rows={}
  for group,cost in costs.items():
   stages=GROUPS.get(group,());roof=sum(model_roof(work.get(s,{}))['roof_us'] for s in stages)
   if group=='MoE':
    combined={k:sum(work.get(s,{}).get(k,0) for s in stages) for k in ('read_bytes','write_bytes','bf16_flops','fp8_flops')};roof=model_roof(combined)['roof_us']
   # Preserve per-role positive excess outside the concurrent MoE envelope.
   excess=max(0.,cost-roof) if group in ('MoE','GPU idle','unattributed busy overlap') else sum(max(0.,audit['critical_us'].get(s,0.)-model_roof(work.get(s,{}))['roof_us']) for s in stages)
   rows[group]=dict(critical_us=cost,roof_us=roof,above_roof_us=excess)
  assert abs(sum(r['critical_us'] for r in rows.values())-audit['wall_us'])<1e-7
  shapes.append(dict(case=c['case'],weight=c['weight'],wall_us=audit['wall_us'],rows=rows,gap_audit=audit))
 loads={}
 for load in LOADS:
  selected=[s for s in shapes if s['case'][:-1]==load];wall=sum(s['weight']*s['wall_us'] for s in selected)/62;groups={g for s in selected for g in s['rows']};rows={}
  for group in groups:
   r={k:sum(s['weight']*s['rows'].get(group,{}).get(k,0.) for s in selected)/62 for k in ('critical_us','roof_us','above_roof_us')};r['above_roof_wall_pct']=100*r['above_roof_us']/wall;rows[group]=r
  loads[load]=dict(wall_us=wall,rows=rows)
 return dict(shapes=shapes,loads=loads)

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+size]);v=shift=0;binary=[]
 for a in raw[4+size:]:
  v|=(a&127)<<shift
  if a&128:shift+=7
  else:binary.append(v);v=shift=0
 assert not shift and len(binary)==x['binary_count']
 for cell in x['cases']:
  ref=cell['dag'];template=x['dag_templates'][ref['template']];origin=ref['id_origin'];cell['dag']=dict(nodes=[[n+origin if n>=0 else n,t] for n,t in template['nodes']],edges=template['edges'])
  unzig=lambda z:(z>>1)^-(z&1)
  for seed in cell['seeds']:
   off=seed.pop('critical_offset');critical=dict(zip(x['stages'],[v/1000 for v in binary[off:off+len(x['stages'])]]));wall=sum(critical.values())
   joins=[];off=seed.pop('joins_offset')
   for i in range(seed.pop('joins_count')):
    row=binary[off+4*i:off+4*(i+1)];ids=[unzig(v) for v in row[:3]]
    ids=[v+origin if v>=0 else v for v in ids];joins.append(dict(zip(('node','winner','runner_up'),ids),slack_us=row[3]/1000))
   seed['account']=dict(wall_us=wall,critical_us=critical,joins=joins)
   counts,keys,causal=seed['fixture'];t=int(cell['case'][1:-1])*(8 if cell['case'].startswith('v') else 1)
   work={k:dict(zip(('read_bytes','write_bytes','bf16_flops','fp8_flops'),v)) for k,v in cell['work_static'].items()};name,fixed,writes,fp8=cell['attention_recipe']
   work[name]=dict(read_bytes=fixed+keys*260,write_bytes=writes,bf16_flops=0 if fp8 else 4096*causal,fp8_flops=4096*causal if fp8 else 0)
   buckets={'moe_small':[c for c in counts if c>0 and (t<=32 or c<=8)],'moe_light':[c for c in counts if t>32 and 8<c<=16],'moe_heavy':[c for c in counts if t>32 and c>16]}
   for name,values in buckets.items():
    experts,pairs=len(values),sum(values)
    work[name]=dict(read_bytes=experts*(8192*4096*9//16+4096*4096*9//16+8+4096*2)+pairs*4096*4,write_bytes=pairs*(4096*6+64*4),bf16_flops=2*pairs*4096*(8192+4096),fp8_flops=0)
   if t<=32:work['moe_small']['read_bytes']+=t*8*4*2;work['moe_small']['write_bytes']+=t*4096*2
   else:work['moe_combine']=dict(read_bytes=sum(counts)*(4096*4+64*4)+t*8*4*2,write_bytes=t*4096*2,bf16_flops=0,fp8_flops=0)
   seed['work']=work
   for key in ('profile_wall_samples_us',):
    off,n=seed[key];scale=1000 if key=='profile_wall_samples_us' else 100;center=round(wall*scale)
    seed[key]=[(center+unzig(v))/scale for v in binary[off:off+n]]
  trace=[];start=0
  for i in range(cell['trace_count']):
   ident,meta,stream,delta,duration=binary[cell['trace_offset']+5*i:cell['trace_offset']+5*(i+1)];start+=delta
   node=dict(x['node_metadata'][meta],node=ident+origin,stream=stream,start_ns=start,end_ns=start+duration);node['name']=x['names'][node['name']];trace.append(node)
  cell['trace']=trace
 return x

def aggregate(x):
 shapes=[shape_summary(c) for c in x['cases']];loads={}
 for load in LOADS:
  selected=[c for c in shapes if c['case'][:-1]==load];assert len(selected)==2
  total=sum(c['weight'] for c in selected);assert total==62
  wall=sum(c['weight']*c['wall_us'] for c in selected)/total
  stages=sorted({s for c in selected for s in c['rows']});rows={}
  for stage in stages:
   fields=next(c['rows'][stage] for c in selected if stage in c['rows'])
   row={k:sum(c['weight']*c['rows'].get(stage,{}).get(k,0.) for c in selected)/total for k in fields}
   row['critical_wall_pct']=100*row['critical_us']/wall;row['above_roof_wall_pct']=100*row['above_roof_us']/wall;rows[stage]=row
  assert abs(sum(r['critical_us'] for r in rows.values())-wall)<max(.002,.02*wall)
  loads[load]=dict(wall_us=wall,rows=rows)
 controls={c['case']:dict(**c['control_event_us'],mapped_over_plain=c['control_event_us']['mapped_event_us']/c['control_event_us']['plain_event_us'],profile_over_mapped=c['control_event_us']['profile_event_us']/c['control_event_us']['mapped_event_us']) for c in x['cases']}
 return dict(shapes=shapes,loads=loads,measurement_controls=controls,empty_event_us=x['empty_event_us'],representative_owners=representative_owners(x['cases']))

def table(report):
 lines=['S15 completion-critical accounting; logical read/FLOP stream roof. Per-stage floors are not additive under overlap.','',
 'Owner table from each shape\'s retained median-seed trace (46/16 weighted). Cells: **critical us / roof us / above-roof % of wall**. Busy logical-DAG gaps go to the owner of the GPU work actually overlapping them. This identifies overlap, not a causal branch. MoE uses one combined unique-work roof across its concurrent branches.','',
 '| Owner | '+' | '.join(LOADS)+' |','|---|'+'---:|'*6]
 for group in ('attention','MoE','projections','glue','GPU idle','unattributed busy overlap'):
  vals=[report['representative_owners']['loads'][load]['rows'].get(group) for load in LOADS]
  if any(v is not None for v in vals):lines.append('| '+group+' | '+' | '.join('—' if v is None else f"{v['critical_us']:.2f} / {v['roof_us']:.2f} / {v['above_roof_wall_pct']:.1f}%" for v in vals)+' |')
 lines+=['','Per-stage logical-DAG accounting below averages all three seeds. **DAG gaps are not launch overhead**: they can contain work on another stream. Use the owner table above to rank stages; branch rows below cannot predict isolated speedups.','', '| Stage | '+' | '.join(LOADS)+' |','|---|'+'---:|'*6]
 stages=('attention_sliding','attention_full','moe_small','moe_heavy','moe_light','moe_meta','moe_combine','qkv_gate','qkv3','gate','o','pool_router','unpooler','unpooler_slice','route','b1','b2','setup','launch_gaps')
 lines+=['Cells: **critical us / roof us / above-roof % of wall**.','']
 # Keep the explanatory line before, not inside, the Markdown table.
 note=lines[-2:];lines=lines[:-2];lines=lines[:2]+note+lines[2:]
 for stage in stages:
  values=[]
  for load in LOADS:
   r=report['loads'][load]['rows'].get(stage)
   values.append('—' if r is None else f"{r['critical_us']:.2f} / {r['roof_us']:.2f} / {r['above_roof_wall_pct']:.1f}%")
  if any(v!='—' for v in values):lines.append('| '+('DAG gaps' if stage=='launch_gaps' else stage)+' | '+' | '.join(values)+' |')
 lines+=['','All-seed logical-DAG above-roof shares (DAG gaps remain unassigned here):']
 for group,stages in GROUPS.items():
  vals=[sum(report['loads'][load]['rows'].get(s,{}).get('above_roof_wall_pct',0) for s in stages) for load in LOADS]
  lines.append('- '+group+': '+' / '.join(f'{v:.1f}%' for v in vals)+' at d1 / d16 / d32 / v1 / v16 / v32.')
 lines+=['','For PLAN “Where the time goes” (representative owner above-roof share):']
 for group in ('attention','MoE','projections','glue','GPU idle'):
  lines.append('- '+group+': '+' / '.join(f"{report['representative_owners']['loads'][load]['rows'][group]['above_roof_wall_pct']:.1f}%" for load in LOADS)+'.')
 lines+=['','All-seed mean wall us/layer: '+' / '.join(f"{report['loads'][load]['wall_us']:.2f}" for load in LOADS)+'.',
 '“DAG gaps” are uncovered intervals along the observed completion chain. Non-kernel event/join nodes carry their predecessor frontier; this is not a CPU launch-overhead counter.',
 'Heavy/light may have zero completion-critical contribution while still contending for bandwidth. Join slack bounds how far the winning branch can improve before another branch becomes critical.']
 lines+=['','Measurement controls (whole replay event us; includes replay/event overhead):','', '| Shape | Plain | Mapped | Profiled | Mapped/plain | Profiled/mapped |', '|---|---:|---:|---:|---:|---:|']
 for case,v in report['measurement_controls'].items():lines.append(f"| {case} | {v['plain_event_us']:.2f} | {v['mapped_event_us']:.2f} | {v['profile_event_us']:.2f} | {v['mapped_over_plain']:.4f} | {v['profile_over_mapped']:.4f} |")
 return '\n'.join(lines)+'\n'

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt',type=Path,help='h200_lab read text or canonical receipt JSON');p.add_argument('--out',type=Path,required=True,help='output directory');a=p.parse_args();text=a.receipt.read_text()
 if a.receipt.suffix=='.json':text=json.loads(text)['result']['log_tail']
 blob=next(s.split('S15_TIMELINE_XZ85 ',1)[1].strip() for s in text.splitlines() if 'S15_TIMELINE_XZ85 ' in s);x=decode(blob);r=aggregate(x);a.out.mkdir(parents=True,exist_ok=True)
 (a.out/'timeline.json').write_text(json.dumps(x,indent=2)+'\n');(a.out/'critical_path.json').write_text(json.dumps(r,indent=2)+'\n');(a.out/'where_time_goes.md').write_text(table(r))
 with (a.out/'critical_path.csv').open('w') as f:
  rows=[dict(load=load,stage=stage,**row) for load,v in r['loads'].items() for stage,row in v['rows'].items()];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('wrote',a.out/'where_time_goes.md')
