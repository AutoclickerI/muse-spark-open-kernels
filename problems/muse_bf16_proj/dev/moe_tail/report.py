"""Retained paired seeds, controls and route/finish timing from the one shell lab."""
import argparse,base64,json,lzma,math,pathlib,statistics,struct

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));n=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+n]);values=[];v=shift=0
 for a in raw[4+n:]:
  v|=(a&127)<<shift
  if a&128:shift+=7
  else:values.append((v>>1)^-(v&1));v=shift=0
 assert not shift and len(values)==x['binary_count']
 for c in x['cases']:
  cols=c['columns'];off=c['sample_offset'];c['samples']=[];c['reference_us']=[]
  for i in range(c['sample_rows']):
   row={};us={}
   for j,arm in enumerate(cols):
    k=off+2*(i*len(cols)+j);row[arm]=1+values[k]*x['ratio_resolution'];us[arm]=c['measured_means'][arm]['a_us']+values[k+1]/100
   c['samples'].append(row);c['reference_us'].append(us)
 return x

def stats(v):return dict(mean=statistics.mean(v),ci95=1.96*statistics.stdev(v)/math.sqrt(len(v)),minimum=min(v),maximum=max(v),n=len(v))
def summary(x):
 result=dict(cases={},loads={},profile_limitation='Small W2 end was not retained by the original profiler label filter; typed finish dependencies and paired timings are retained.')
 for c in x['cases']:
  r={a:stats([s[a] for s in c['samples']]) for a in c['columns']};result['cases'][c['case']]=dict(paired=r,gates=c['gates'],aa_schedule_identical=c['aa_schedule_identical'],measured_means=c['measured_means'],profiles=c['profiles'])
 for batch in (16,32):
  cs=[next(c for c in x['cases'] if c['case']=='v'+str(batch)+w) for w in ('S','F')];rows={}
  for arm in cs[0]['columns']:
   values=[]
   for i,(a,b) in enumerate(zip(cs[0]['samples'],cs[1]['samples'])):
    weights=[k*c['reference_us'][i][arm] for c,k in zip(cs,(46,16))];values.append((weights[0]*a[arm]+weights[1]*b[arm])/sum(weights))
   rows[arm]=stats(values)
  result['loads']['v'+str(batch)]=rows
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();t=a.receipt.read_text()
 if a.receipt.suffix=='.json':t=json.loads(t)['result']['log_tail']
 blob=next(l.split('MOE_TAIL_XZ85 ',1)[1].strip() for l in t.splitlines() if 'MOE_TAIL_XZ85 ' in l);x=decode(blob);a.out.write_text(json.dumps(x,indent=2)+'\n');a.out.with_suffix('.summary.json').write_text(json.dumps(summary(x),indent=2)+'\n');print(json.dumps(summary(x)['loads'],indent=2))
