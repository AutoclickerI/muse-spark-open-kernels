"""Decode every retained observation and report paired whole-layer ratios and role tails."""
import argparse,base64,json,lzma,math,statistics,struct
from pathlib import Path

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+size]);binary=struct.unpack('<'+'I'*x['binary_count'],raw[4+size:])
 for case in x['cases']:
  cols=case['columns'];off=case['sample_offset'];samples=[]
  for i in range(case['sample_rows']):
   row=dict(zip(cols,binary[off+i*len(cols):off+(i+1)*len(cols)]));a=row['A']
   samples.append({n:(v if n=='A' else a+((v>>1)^-(v&1)))/1000 for n,v in row.items()})
  case['samples']=samples
  for arm,p in case['profiles'].items():
   origin=case['dag'][arm]['id_origin']
   for row in case['dag'][arm]['nodes']:
    if row[0]>=0:row[0]+=origin
   nodes=[];start=0
   for ident,meta,stream,delta,duration in p['nodes']:
    start+=delta;node=dict(x['node_metadata'][meta],node=ident+origin,stream=stream,start_ns=start,end_ns=start+duration)
    node['name']=x['node_names'][node['name']];nodes.append(node)
   p['nodes']=nodes
 return x

def stats(values):return dict(mean=statistics.mean(values),ci95=1.96*statistics.stdev(values)/math.sqrt(len(values)))

def summary(x):
 out=dict(cases={},load={})
 for cell in x['cases']:
  samples=cell['samples'];pairs={n:stats([s[n]/s['A'] for s in samples]) for n in samples[0] if n!='A'}
  r=out['cases'][cell['case']]=dict(gates=cell['gates'],paired=pairs,profiles={},ready={role:{k:statistics.median(v) if isinstance(v,list) else v for k,v in a.items()} for role,a in cell['ready'].items()})
  for arm,p in cell['profiles'].items():
   nodes=p['nodes'];stages={s:dict(start=min(n['start_ns'] for n in nodes if n['stage']==s)/1000,end=max(n['end_ns'] for n in nodes if n['stage']==s)/1000) for s in {n['stage'] for n in nodes}}
   prior=0.;increments={}
   for s in ('init','qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler_slice','b2'):
    if s in stages:increments[s]=max(0.,stages[s]['end']-prior);prior=max(prior,stages[s]['end'])
   roles={}
   for role,producer in [('b1','o'),('route','pool_router'),('b2','unpooler_slice')]:
    b,a=stages[role],stages[producer];roles[role]=dict(residence=b['end']-b['start'],completion_increment=increments[role],overlap_before_producer_end=max(0.,min(b['end'],a['end'])-b['start']),tail_after_producer_end=max(0.,b['end']-max(b['start'],a['end'])))
   dag=cell['dag'][arm];index={n['node']:n for n in nodes};pred={}
   for role in roles:
    targets={i for i,n in enumerate(dag['nodes']) if index.get(n[0],{}).get('stage')==role}
    pred[role]=[dict(source=index.get(dag['nodes'][a][0],{'type':dag['nodes'][a][1]}),type=t,from_port=fp,to_port=tp) for a,b,t,fp,tp in dag['edges'] if b in targets]
   r['profiles'][arm]=dict(roles=roles,stages=stages,increments=increments,dag_predecessors=pred,nonkernel_nodes=[n for n in dag['nodes'] if n[1]!=0])
 if len(x['cases'])==2:
  s,f=x['cases']
  for n in s['samples'][0]:
   if n=='A':continue
   out['load'][n]=stats([(46*a[n]+16*b[n])/(46*a['A']+16*b['A']) for a,b in zip(s['samples'],f['samples'])])
 return out

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 blob=next(s.split('D1_CHAIN_XZ85 ',1)[1].strip() for s in a.receipt.read_text().splitlines() if 'D1_CHAIN_XZ85 ' in s);x=decode(blob)
 a.out.write_text(json.dumps(x,indent=2)+'\n');a.out.with_suffix('.summary.json').write_text(json.dumps(summary(x),indent=2)+'\n');print('decoded',len(x['cases']),'cases',len(x['seeds']),'seeds',len(x['cases'][0]['samples'][0]),'arms')
