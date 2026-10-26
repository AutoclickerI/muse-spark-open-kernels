"""Held-out six-load G0, preserving paired seeds and sliding/full covariance."""
import argparse,base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines

def ci(values):
 z=[math.log(x) for x in values];mu=statistics.mean(z);h=(2.364624251 if len(z)==8 else 2.131449546)*statistics.stdev(z)/math.sqrt(len(z));return [math.exp(mu),math.exp(mu-h),math.exp(mu+h)]

def report(path):
 d=json.loads(path.read_text());tail=d['result']['log_tail'];lines=result_lines(tail);blobs=[s.split(' ',1)[1] for s in lines if s.startswith('WHOLE_XZ85 ')];assert len(blobs)==1,lines
 raw=json.loads(lzma.decompress(base64.b85decode(blobs[0])));errors=[e for p in read_packets(tail) for e in p['packet'].get('transport_errors',[])];rows={tuple(row['case']):row['values'] for row in raw['rows']};arms=list(next(iter(rows.values())));per={};loads={};global_r={}
 for key,values in rows.items():per[str(key)]={arm:ci([b/a for a,b in zip(values['A'],values[arm])]) for arm in arms if arm!='A'}
 for b,q,name in ((1,1,'decode1'),(16,1,'decode16'),(32,1,'decode32'),(1,8,'verify1'),(16,8,'verify16'),(32,8,'verify32')):
  s,f=rows[b,q,2048],rows[b,q,0];a=[46*x+16*y for x,y in zip(s['A'],f['A'])];load={}
  for arm in arms:
   if arm=='A':continue
   rv=[aa/(46*x+16*y) for aa,x,y in zip(a,s[arm],f[arm])];load[arm]=dict(R=ci(rv),samples=rv)
  loads[name]=load
 for arm in arms:
  if arm=='A':continue
  rv=[math.exp(statistics.mean(math.log(load[arm]['samples'][i]) for load in loads.values())) for i in range(len(raw['seeds']))];global_r[arm]=dict(R=ci(rv),samples=rv,G0=raw['complete'] and not errors and ci(rv)[1]>1)
 out=dict(job=d['id'],receipt=str(path.relative_to(R)),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),valid=raw['complete'] and not errors,transport_errors=errors,raw=raw,per_shape_candidate_control_time=per,loads=loads,arms=global_r,CI='Student-t'+str(len(raw['seeds'])-1)+' over paired log ratios. Each seed retains six-load covariance; Tload=46*sliding+16*full. Fresh seeds/one heldout weight+router seed.',limits=['Fixed candidate composition on frozen S19; no standalone projection-board residual attribution','Official scored decode validation and P4 final integration remain separate'])
 (H/'whole_results.json').write_text(json.dumps(out,indent=2)+'\n')
 for arm,v in global_r.items():print(arm,*[round(x,8) for x in v['R']],'G0',v['G0'])
 for key,v in per.items():print(key,{a:[round(x,5) for x in r] for a,r in v.items()})
 print('PREPARE',raw['prepare_s'],raw['added_prepare']);return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('receipt');a=p.parse_args();path=Path(a.receipt)
 if not path.exists():path=next((R/'runs').glob('muse_*/*'+a.receipt+'.json'))
 report(path.resolve())
