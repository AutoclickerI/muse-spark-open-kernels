"""Decode checksummed oracle evidence; PRE/frozen-route arms are never candidates."""
import argparse,base64,json,math,re,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[5];sys.path.insert(0,str(ROOT/'tools'))
from lab_packet import result_lines,read_packets
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();r=json.loads(a.receipt.read_text());native='result'in r
raw=r['result'].get('log_tail','') if native else '\n'.join(r['lines']);packets=read_packets(raw);tail='\n'.join(result_lines(raw))
assert 'LAB_ERROR' not in tail and 'LAB_PACKET_ERROR' not in tail and 'LAB_PACKET_OVERSIZE' not in tail
z=re.search(r'GATE L2/row=([\d.eE+-]+)/([\d.eE+-]+) END seeds=(\d+)',tail);assert z and float(z[1])<=.015 and float(z[2])<=.075
arc=json.loads(re.search(r'^SEEDS (.*)$',tail,re.M)[1]);ns=arc['n'];assert ns==int(z[3]);data=struct.unpack('<'+'f'*(2*ns*25),base64.b85decode(arc['data']));rows=[[[data[(mi*ns+s)*25+i]for i in range(25)]for s in range(ns)]for mi in range(2)];t={3:3.182446,7:2.364624,11:2.200985,15:2.13145}.get(ns-1,1.96)
def summary(x):
 mu=statistics.mean(x);se=statistics.stdev(x)/math.sqrt(len(x));return dict(mean=mu,SE=se,CI95=[mu-t*se,mu+t*se],seeds=len(x),samples=x)
out=dict(receipt=str(a.receipt),native_H200=native,packet_checksums=[p['checksum']for p in packets],gate=dict(L2=float(z[1]),row=float(z[2])),arms=[],contrasts=[],AA={},resources=next(l[4:]for l in tail.splitlines()if l.startswith('RES ')),controls='Route metadata generated once per fresh fixture outside timing and copied identically to all arms; PRE gather outside timing, coldL2; no arm nominatable as submitted.')
for mi,m in enumerate((128,256)):
 rr=rows[mi]
 for i,arm in enumerate(arc['names']):
  spec=dict(M=m,arm=arm,ratio=summary([r[i]for r in rr]),us=summary([r[16+i]for r in rr]),promotable=False)
  out['arms'].append(spec);print(m,arm,spec['ratio']['mean'],spec['ratio']['CI95'])
 for i,(new,old) in enumerate(arc['pairs']):out['contrasts'].append(dict(M=m,new=new,old=old,ratio=summary([r[9+i]for r in rr]),pairing='Direct cold ABBA, same seed/fixture'))
 dif=[r[15]-r[14]for r in rr];out['AA'][m]=dict(signed_mean_us=statistics.mean(dif),absolute_mean_us=statistics.mean(abs(x)for x in dif),maximum_seed_mean_us=max(abs(x)for x in dif),ratio=summary([r[15]/r[14]for r in rr]))
pre=[next(x for x in out['arms']if x['M']==m and x['arm']=='PREPACK')['ratio']for m in(128,256)]
out['size_case']=dict(free_gather_PACK_point_ceiling_R=math.exp(-sum(math.log(x['mean'])for x in pre)/6),optimistic_CI_endpoint_R=math.exp(-sum(math.log(x['CI95'][0])for x in pre)/6),assumption='Onlyverify16/32 change; assume MoE equals the entire layer at each, all other stages free. Actual score transfer is smaller. CI endpoint calculation is an optimistic size illustration, not joint95confidence.',required_equal_MoE_reduction_for_R101_even_at_full_layer_share=1-1/(1.01**3),decision='Propose parking P3b by current measured decision value; no supported >1pct six-load next build. Theoretical31pct gap ceiling is not disproven.')
(H/('results.json'if native else'local_results.json')).write_text(json.dumps(out,indent=2)+'\n')
for x in out['contrasts']:print('PAIR',x['M'],x['new']+'/'+x['old'],x['ratio']['mean'],x['ratio']['CI95'])
print(out['size_case'])
