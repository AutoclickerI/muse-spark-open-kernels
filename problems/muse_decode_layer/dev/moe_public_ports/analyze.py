"""Decode all exact rounds; CI unit is the independent fresh input seed."""
import argparse,base64,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'));from lab_packet import result_lines,read_packets

def decode(tail):
 assert all(not r['packet'].get('transport_errors') for r in read_packets(tail)), 'transport device fault'
 line=next(l for l in result_lines(tail) if l.startswith('RESULT_BIN85 '));raw=lzma.decompress(base64.b85decode(line.split(' ',1)[1]));n=struct.unpack('<I',raw[:4])[0];d=json.loads(raw[4:4+n]);binary=raw[4+n:]
 def integer(at):
  v=0;b=0
  while True:
   x=binary[at];at+=1;v|=(x&127)<<b
   if x<128:return v,at
   b+=7
 for field in ('round_ns','phase_ns'):
  values={}
  for M,spec in d[field].items():
   at=spec['offset'];rows=[]
   for _ in range(spec['seeds']):
    row={}
    for key in spec['keys']:
     first,at=integer(at);vs=[first]
     for _ in range(spec['rounds']-1):
      delta,at=integer(at);vs.append(first+(delta//2 if delta%2==0 else -(delta+1)//2))
     row[key]=vs
    rows.append(row)
   values[M]=rows
  d[field]=values
 return d

def ci(values):
 c=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values));e=2.364624251*se
 return dict(r=c,lo=c-e,hi=c+e,se=se,n=len(values),seed_ratios=values)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();receipt=json.loads(a.receipt.read_text());d=decode(receipt['result']['log_tail']);manifest=json.loads((H/'manifest.json').read_text())
 assert d['complete'] and not d['failures'];assert d['seeds']==manifest['seeds'];assert d['identities']==manifest['identities'];assert set(d['round_ns'])=={'1','8','16','32','128','256'}
 fixed=d['fixed_ns'];whole={};phases={};aa={};selection={};mechanisms=[]
 for M,rows in d['round_ns'].items():
  assert len(rows)==8 and all(len(v)==4 and min(v)>fixed for row in rows for v in row.values())
  means=[{k:statistics.mean(v)-fixed for k,v in row.items()} for row in rows];aa[M]=ci([r['A2']/r['A'] for r in means]);whole[M]={}
  for name in rows[0]:
   if name in ('A','A2'):continue
   whole[M][name]=dict(**ci([r[name]/r['A'] for r in means]),mean_us=statistics.mean(r[name] for r in means)/1000,base_us=statistics.mean(r['A'] for r in means)/1000)
  best=min(whole[M],key=lambda n:whole[M][n]['r']);selection[M]=best if whole[M][best]['r']<1 else 'A'
  prows=d['phase_ns'][M];pmeans=[{k:statistics.mean(v)-fixed for k,v in row.items()} for row in prows];phases[M]={}
  for key in prows[0]:
   name,stage=key.split(':');den='A:'+stage
   q=dict(mean_us=statistics.mean(r[key] for r in pmeans)/1000,base_us=statistics.mean(r[den] for r in pmeans)/1000 if den in prows[0] else None,comparison='isolated ready fixture; S19 small-M phases unavailable' if int(M)<=32 else 'isolated matched projection envelope; whole S19 graph unchanged')
   if name!='A' and den in prows[0]:q.update(ci([r[key]/r[den] for r in pmeans]))
   phases[M][key]=q
   if q.get('r',2)<1:mechanisms.append(dict(M=int(M),key=key,borrowable=False,reason='Native phase boundaries differ; see ABI_H200.md',**q))
  print('M'+M,'AA',f"{aa[M]['r']:.5f}[{aa[M]['lo']:.5f},{aa[M]['hi']:.5f}]",*(n+':'+f"{q['r']:.5f}[{q['lo']:.5f},{q['hi']:.5f}]" for n,q in whole[M].items()))
 result=dict(receipt=str(a.receipt.resolve().relative_to(R)),job=receipt.get('job_id',receipt.get('id',receipt.get('job'))),whole=whole,phases=phases,AA=aa,selection=selection,mechanisms=mechanisms,checks=d['checks'],elapsed_s=d['elapsed_s'],fixed_ns=fixed,seeds=d['seeds'],weight_seed=d['weight_seed'],identities=d['identities'],method='Mean all 4 cold rounds per seed; arithmetic mean paired port/S19 seed ratio, two-sided95% Student-t7; no seed/round excluded. Eight input/route seeds, one fresh weight bank. P4 owns held-out weight/whole-layer G0.',promotion=False)
 (H/'receipt_data.json').write_text(json.dumps(d,indent=2)+'\n');(H/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Public MoE ports versus exact S19','','Time ratio port/S19 (lower is faster); 95% paired-seed t7 CIs. Every round retained.','', '| M | operator O | Lee X1 fused | Lee X2 fused | Lee S1/F1 | Lee S2/F2 | pick |','|---:|---:|---:|---:|---:|---:|---|']
 fmt=lambda q:f"{q['r']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]"
 for M in (1,8,16,32,128,256):
  qs=whole[str(M)];k='F' if M<=32 else 'S';lines.append('| '+str(M)+' | '+' | '.join(fmt(qs[n]) for n in ('O','X1','X2',k+'1',k+'2'))+' | '+selection[str(M)]+' |')
 lines+=['','## Isolated phases','','Native port phases have different contracts. O W2 excludes inline norm/SS; Lee finish reads BF16 Y/scalar SS. Their ratios are diagnostic; **ABI_H200.md supplies matched-work comparisons**. The small-M S19 body fuses both projections and finish: no separate denominator exists. Fused Lee F has on-chip intermediates and is likewise inseparable. Values below time explicit X split ports, including W2 scratch clearing; they are checked as complete MoE pipelines. Large-M S19 stages retain all three expert branches on their original streams, with a barrier between isolated projection envelopes. Whole-MoE timing retains S19’s original overlap.','','| M | arm:phase | us | port/S19 CI |','|---:|---|---:|---:|']
 for M in (1,8,16,32,128,256):
  for key,q in phases[str(M)].items():lines.append(f"| {M} | {key} | {q['mean_us']:.3f} | {fmt(q) if 'r' in q else 'N/A'} |")
 lines+=['',f"Receipt: `{result['receipt']}`; checks `{d['checks']}`; elapsed {d['elapsed_s']:.1f}s; fixed {fixed:.1f}ns.", '',result['method'],'']
 (H/'H200.md').write_text('\n'.join(lines))
if __name__=='__main__':main()
