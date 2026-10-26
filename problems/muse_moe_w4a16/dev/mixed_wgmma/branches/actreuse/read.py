import argparse,base64,json,math,re,statistics,struct
from pathlib import Path
H=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();receipt=json.loads(a.receipt.read_text());tail='\n'.join(receipt['lines']) if 'lines'in receipt else receipt['result'].get('log_tail','')
assert 'LAB_ERROR' not in tail and 'LAB_TRUNCATED' not in tail
z=re.search(r'GATE L2/row=([\d.eE+-]+)/([\d.eE+-]+) END seeds=(\d+)',tail);assert z and float(z[1])<=.015 and float(z[2])<=.075
arc=json.loads(re.search(r'^SEEDS (.*)$',tail,re.M)[1]);ns=arc['n'];assert ns==int(z[3]);data=struct.unpack('<'+'f'*(2*ns*23),base64.b85decode(arc['data']));rows=[[[data[(mi*ns+s)*23+i] for i in range(23)] for s in range(ns)] for mi in range(2)]
t={3:3.182446,4:2.776445,7:2.364624,11:2.200985,15:2.13145}.get(ns-1,1.96)
def summary(x):
 mu=statistics.mean(x);se=statistics.stdev(x)/math.sqrt(len(x));return dict(mean=mu,SE=se,CI95=[mu-t*se,mu+t*se],seeds=len(x),samples=x)
out=dict(receipt=str(a.receipt),native_H200='result'in receipt,gate=dict(L2=float(z[1]),row=float(z[2])),arms=[],AA={},resources=next(l[4:] for l in tail.splitlines() if l.startswith('RES ')))
for mi,m in enumerate((128,256)):
 rr=rows[mi];dif=[v[8]-v[7] for v in rr];out['AA'][m]=dict(signed_mean_us=statistics.mean(dif),absolute_mean_us=statistics.mean(abs(x) for x in dif),max_seed_mean_us=max(abs(x) for x in dif),ratio=summary([v[8]/v[7] for v in rr]))
 for j,arm in enumerate(arc['names']):
  r=dict(M=m,arm=arm,ratio=summary([v[j] for v in rr]),activation_MB=summary([v[9+j] for v in rr]),weight_MB=summary([v[16+j] for v in rr]));r['nominate_to_P4']=out['native_H200'] and j>0 and r['ratio']['mean']<1;out['arms'].append(r)
  print(m,arm,r['ratio']['mean'],r['ratio']['CI95'],r['activation_MB']['mean'],'P4',r['nominate_to_P4'])
(H/('results.json' if out['native_H200'] else 'local_results.json')).write_text(json.dumps(out,indent=2)+'\n')
