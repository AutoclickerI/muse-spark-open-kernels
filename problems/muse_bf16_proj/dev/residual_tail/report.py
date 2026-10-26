"""Per-shape point nominations and diagnostic six-load estimates; P4 confirms on held-out seeds."""
import argparse,base64,json,lzma,math,pathlib,statistics,struct

def decode(blob):
 raw=lzma.decompress(base64.b85decode(blob));n=struct.unpack_from('<I',raw)[0];x=json.loads(raw[4:4+n]);data=struct.unpack('<'+'I'*x['binary_count'],raw[4+n:])
 for c in x['cases']:
  k=c['sample_offset'];c['samples']=[]
  for i in range(c['sample_rows']):
   row={}
   for arm in c['columns']:
    a,b=data[k:k+2];k+=2;row[arm]=dict(a_us=a/1000,b_us=b/1000,ratio=b/a)
   c['samples'].append(row)
 return x

def stats(v):return dict(mean=statistics.mean(v),se=statistics.stdev(v)/math.sqrt(len(v)),n=len(v),min=min(v),max=max(v))
def summary(x):
 shapes={c['case']:{a:stats([row[a]['ratio'] for row in c['samples']]) for a in c['columns']} for c in x['cases']}
 nominations=[dict(shape=k,arm=a,**v,seeds=x['seeds'],receipt_base=x['base_sha256']) for k,r in shapes.items() for a,v in r.items() if a!='A2' and v['mean']<1 and not next(c for c in x['cases'] if c['case']==k)['failures'].get(a)]
 scores={}
 for arm in (a for a in x['arms'] if a not in ('A','A2')):
  loads=[]
  for prefix in ('d1','v1'):
   s,f=[next(c for c in x['cases'] if c['case']==prefix+w) for w in ('S','F')]
   loads.append([(46*a[arm]['a_us']+16*b[arm]['a_us'])/(46*a[arm]['b_us']+16*b[arm]['b_us']) for a,b in zip(s['samples'],f['samples'])])
  rr=[math.exp((math.log(a)+math.log(b))/6) for a,b in zip(*loads)];scores[arm]=dict(**stats(rr),interpretation='Other four loads use identical S15 code; selection-data diagnostic, not held-out promotion')
 return dict(shapes=shapes,nominations=nominations,score_diagnostics=scores,failures={c['case']:c['failures'] for c in x['cases']},task_checks=sum(c['task_checks'] for c in x['cases']),profiles={c['case']:c['profiles'] for c in x['cases']},seeds=x['seeds'],base_sha256=x['base_sha256'],runtime_sha256=x['runtime_sha256'],elapsed_s=x['elapsed_s'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt',type=pathlib.Path);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();t=a.receipt.read_text()
 if a.receipt.suffix=='.json':t=json.loads(t)['result']['log_tail']
 blob=next(l.split('RESIDUAL_TAIL_XZ85 ',1)[1].strip() for l in t.splitlines() if 'RESIDUAL_TAIL_XZ85 ' in l);x=decode(blob);a.out.parent.mkdir(exist_ok=True,parents=True);a.out.write_text(json.dumps(x,indent=2)+'\n');result=summary(x);result['receipt']=str(a.receipt);a.out.with_suffix('.summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
