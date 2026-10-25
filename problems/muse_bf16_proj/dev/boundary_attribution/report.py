"""Decode a retained BD_BATCH_XZ85 line and report paired ratios plus DAG predecessors."""
import argparse,base64,json,lzma,math,statistics,struct
from pathlib import Path

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));n=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+n]);binary=struct.unpack('<'+'I'*x['binary_count'],raw[4+n:])
 for cell in x['confirmation']['cells']+x['boundaries']:
  names=cell['sample_columns'];off=cell['sample_offset'];samples=[]
  for i in range(cell['sample_rows']):
   row=dict(zip(names,binary[off+i*len(names):off+(i+1)*len(names)]));baseline=row['A']
   samples.append({n:(v if n=='A' else baseline+((v>>1)^-(v&1)))/1000 for n,v in row.items()})
  cell['samples']=samples
 for cell in x['boundaries']:
  for arm,p in cell['profiles'].items():
   dag=cell['dag'][arm];origin=dag['id_origin']
   for row in dag['nodes']:
    if row[0]>=0:row[0]+=origin
   nodes=[];previous_start=0
   for ident,meta,stream,start_delta,duration in p['nodes']:
    start=previous_start+start_delta;end=start+duration;previous_start=start
    node=dict(x['node_metadata'][meta],node=ident+origin,stream=stream,start=start/1000,end=end/1000)
    node['name']=x['names'][node['name']];nodes.append(node)
   p['nodes']=nodes
   stages={s:dict(start=min(n['start'] for n in nodes if n['stage']==s),end=max(n['end'] for n in nodes if n['stage']==s)) for s in {n['stage'] for n in nodes}}
   previous=0.;increments={}
   for s in ('init','qkv3','qkv_gate','attn','gate','o','b1','pool_router','route','moe','unpooler','b2'):
    if s in stages:increments[s]=max(0.,stages[s]['end']-previous);previous=max(previous,stages[s]['end'])
   bounds={}
   for role,producer in (('b1','o'),('b2','unpooler')):
    b,r=stages[role],stages[producer]
    bounds[role]=dict(residence=b['end']-b['start'],increment=increments[role],before_producer_end=max(0.,min(b['end'],r['end'])-b['start']),after_producer_end=max(0.,b['end']-max(b['start'],r['end'])),start_minus_producer_end=b['start']-r['end'])
   p.update(stages=stages,increments=increments,boundaries=bounds)
 return x

def summary(x):
 out={'confirmation':{},'boundaries':{}}
 for section,cells in [('confirmation',x['confirmation']['cells']),('boundaries',x['boundaries'])]:
  for cell in cells:
   samples=cell['samples'];ratios={}
   for arm in samples[0]:
    if arm=='A':continue
    v=[s[arm]/s['A'] for s in samples];ratios[arm]=dict(latency_ratio=statistics.mean(v),ci95=1.96*statistics.stdev(v)/math.sqrt(len(v)))
   r=out[section][cell['case']]=dict(gates=cell['gates'],paired=ratios)
   if section=='boundaries':
    r['footprints']=cell['footprints'];r['ready']={role:{k:statistics.median(v) for k,v in val.items()} for role,val in cell['ready'].items()}
    r['profiles']={}
    for arm,p in cell['profiles'].items():
     dag=cell['dag'][arm];by={n['node']:n for n in p['nodes']};pred={}
     for role in ('b1','b2','unpooler','moe'):
      indices={i for i,n in enumerate(dag['nodes']) if by.get(n[0],{}).get('stage')==role}
      pred[role]=[dict(source=by.get(dag['nodes'][a][0],{'node':dag['nodes'][a][0],'type':dag['nodes'][a][1]}),edge_type=typ,from_port=fp,to_port=tp) for a,b,typ,fp,tp in dag['edges'] if b in indices]
     r['profiles'][arm]=dict(boundaries=p['boundaries'],increments=p['increments'],immediate_predecessors=pred,nonkernel_nodes=[n for n in dag['nodes'] if n[1]!=0])
 return out

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 text=a.receipt.read_text();blob=next(line.split('BD_BATCH_XZ85 ',1)[1].strip() for line in text.splitlines() if 'BD_BATCH_XZ85 ' in line)
 x=decode(blob);a.out.write_text(json.dumps(x,indent=2)+'\n');a.out.with_suffix('.summary.json').write_text(json.dumps(summary(x),indent=2)+'\n');print('decoded',len(x['confirmation']['cells']),len(x['boundaries']),'fixtures')
