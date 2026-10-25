"""Decode the complete rebuild receipt and compare whole-layer paired ratios with role attribution."""
import argparse,base64,json,lzma,math,statistics,struct
from pathlib import Path

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));size=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+size]);data=raw[4+size:];binary=[];v=shift=0
 for a in data:
  v|=(a&127)<<shift
  if a&128:shift+=7
  else:binary.append(v);v=shift=0
 assert shift==0 and len(binary)==x['binary_count']
 for cell in x['cases']:
  cols=cell['columns'];off=cell['sample_offset'];samples=[]
  for i in range(cell['sample_rows']):
   row=dict(zip(cols,binary[off+i*len(cols):off+(i+1)*len(cols)]));a=row['A']/100;samples.append({n:a if n=='A' else a*(1+((z>>1)^-(z&1))/10000) for n,z in row.items()})
  cell['samples']=samples
  for arm,p in cell['profiles'].items():
   ref=cell['dag'][arm];origin=ref['id_origin'];template=x['dag_templates'][ref['template']]
   cell['dag'][arm]=dict(id_origin=origin,nodes=[list(n) for n in template['nodes']],edges=[list(e) for e in template['edges']])
   for row in cell['dag'][arm]['nodes']:
    if row[0]>=0:row[0]+=origin
   nodes=[];start=0
   packed=[binary[p['node_offset']+i*5:p['node_offset']+(i+1)*5] for i in range(p['node_count'])]
   del p['node_offset'],p['node_count']
   for ident,meta,stream,delta,duration in packed:
    start+=delta;node=dict(x['node_metadata'][meta],node=ident+origin,stream=stream,start_ns=start,end_ns=start+duration);node['name']=x['node_names'][node['name']];nodes.append(node)
   p['nodes']=nodes
 return x

def stats(v):return dict(mean=statistics.mean(v),ci95=1.96*statistics.stdev(v)/math.sqrt(len(v)))

def summary(x):
 result=dict(cases={},loads={})
 for cell in x['cases']:
  samples=cell['samples'];r=result['cases'][cell['case']]=dict(gates=cell['gates'],quality=cell['quality'],ready=cell['ready'],paired={n:stats([s[n]/s['A'] for s in samples]) for n in samples[0] if n!='A'},profiles={})
  for arm,p in cell['profiles'].items():
   nodes=p['nodes'];stages={s:dict(start=min(n['start_ns'] for n in nodes if n['stage']==s)/1000,end=max(n['end_ns'] for n in nodes if n['stage']==s)/1000) for s in {n['stage'] for n in nodes}}
   stages.update({s:dict(start=v['start_ns']/1000,end=v['end_ns']/1000) for s,v in p['stage_envelopes'].items()})
   prior=0.;increments={}
   for s in ('init','qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler','b2'):
    if s in stages:increments[s]=max(0.,stages[s]['end']-prior);prior=max(prior,stages[s]['end'])
   roles={}
   for role,producer in [('b1','o'),('b2','unpooler'),('route','pool_router')]:
    b,a=stages[role],stages[producer];roles[role]=dict(residence=b['end']-b['start'],completion_increment=increments[role],start_minus_producer_end=b['start']-a['end'],overlap_before_producer_end=max(0.,min(b['end'],a['end'])-b['start']),tail_after_producer_end=max(0.,b['end']-max(b['start'],a['end'])))
   dag=cell['dag'][arm];by={n['node']:n for n in nodes};pred={}
   for role in roles:
    targets={i for i,n in enumerate(dag['nodes']) if by.get(n[0],{}).get('stage')==role}
    pred[role]=[dict(source=by.get(dag['nodes'][a][0],{'type':dag['nodes'][a][1],'stage':dag['nodes'][a][2]}),type=t,from_port=fp,to_port=tp) for a,b,t,fp,tp in dag['edges'] if b in targets]
   r['profiles'][arm]=dict(roles=roles,stages=stages,increments=increments,dag_predecessors=pred,nonkernel_nodes=[n for n in dag['nodes'] if n[1]!=0])
 for load in sorted({c['case'][:-1] for c in x['cases']}):
  s,f=[next(c for c in x['cases'] if c['case']==load+w) for w in ('S','F')]
  result['loads'][load]={n:stats([(46*a[n]+16*b[n])/(46*a['A']+16*b['A']) for a,b in zip(s['samples'],f['samples'])]) for n in s['samples'][0] if n!='A'}
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();blob=next(s.split('BR_REBUILD_XZ85 ',1)[1].strip() for s in a.receipt.read_text().splitlines() if 'BR_REBUILD_XZ85 ' in s);x=decode(blob)
 a.out.write_text(json.dumps(x,indent=2)+'\n');a.out.with_suffix('.summary.json').write_text(json.dumps(summary(x),indent=2)+'\n');print('decoded',len(x['cases']),'shapes',len(x['seeds']),'seeds')
