"""Decode exact native batch, verify identities/gates, report all paired points."""
import argparse,base64,hashlib,json,lzma,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain';sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();receipt=json.loads(a.receipt.read_text());tail=receipt.get('result',{}).get('log_tail','')
meta=json.loads((o/'lab_manifest.json').read_text());packets=read_packets(tail)
for p in packets:
 if p['name']=='chain_attention':
  assert p['source_sha256']==meta['payload_sha256'][:16]
  assert not p['packet'].get('transport_errors'),p['packet']
lines=result_lines(tail);rows=[x[12:] for x in lines if x.startswith('RESULT_XZ85 ')]
if not rows:
 out=dict(measured=False,job=receipt['id'],reason='No complete native TASK-gated archive',errors=[x for x in lines if x.startswith('LAB_ERROR')]);(o/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(out);raise SystemExit(1)
assert len(rows)==1
d=json.loads(lzma.decompress(base64.b85decode(rows[0])));assert d['complete'] and not d['failures']
assert d['counts']=={k:meta[k+'_TASK'] for k in ('official','stress','scored','partial')},d['counts']
assert d['base_sha256']==meta['base_sha256'] and list(d['seeds'])==meta['seeds']
assert d['profiles'] and d['profile_names'],'No native profiling proof'
assert all(n>0 for n in d['graph_nodes'].values())
queue=json.loads((o/'queue.json').read_text());assert hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()==queue['wrapper_sha256']
def ratio(samples,n,c='A'):
 rr=[s[n]/s[c] for s in samples];assert min(rr)>0
 r=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5
 return dict(r=r,se=se,lo=r-2.364624251*se,hi=r+2.364624251*se)
summary={};points=[];pairs={};profiles={}
for shape,samples in d['seed_us'].items():
 assert len(samples)==8 and all(v>0 for s in samples for v in s.values())
 summary[shape]=dict(control_us=statistics.mean(s['A'] for s in samples),ratios={n:ratio(samples,n) for n in samples[0] if n!='A'})
 for n in samples[0]:
  if n not in ('A','A2'):
   changed=not(n=='S' and shape not in ('16x8s','1x8f'))
   control_alias=n.startswith('D') or n=='S'
   points.append(dict(shape=shape,arm=n,**ratio(samples,n),changed=changed,control_alias=control_alias,nominated=changed and not control_alias and ratio(samples,n)['r']<1,seeds=d['seeds']))
 for n in samples[0]:
  if n.startswith('R') and n[1:].split('w')[0].isdigit():
   c='M'+n[1:].split('w')[0]
   if c in samples[0]:pairs[shape+':'+n+'/'+c]=ratio(samples,n,c)
  if n.startswith('F') and 'G'+n[1:] in samples[0]:pairs[shape+':'+n+'/G'+n[1:]]=ratio(samples,n,'G'+n[1:])
  if n.startswith('D') and 'M'+n[1:] in samples[0]:pairs[shape+':'+n+'/M'+n[1:]]=ratio(samples,n,'M'+n[1:])
for shape,rows in d['profiles'].items():
 profiles[shape]=[dict(name=d['profile_names'][v[0]],start_us=v[1]/1000,end_us=v[2]/1000,grid=v[3:6],block=v[6],smem=v[7],regs=v[8]) for v in rows]
launches={}
for shape in d['seed_us']:
 launches[shape]={n:d['graph_nodes'][shape+':'+n] for n in d['seed_us'][shape][0]}
assert launches['1x8s']['A']==launches['1x8s']['S']==1
for shape in ('16x8s','1x8f'):
 assert launches[shape]['A']==5 and launches[shape]['S']==4,launches[shape]
 assert not any('prepare_varlen_num_blocks' in x['name'] for x in profiles[shape+':S'])
for shape in ('16x8f','32x8f'):
 for cs in (4,8):assert launches[shape]['F'+str(cs)]==1 and launches[shape]['G'+str(cs)]==2,launches[shape]
preferred={s:min((p for p in points if p['shape']==s and p['nominated']),key=lambda p:p['r'])['arm'] for s in summary if any(p['shape']==s and p['nominated'] for p in points)}
out=dict(measured=True,job=receipt['id'],receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),result=d,summary=summary,points=points,pairs=pairs,preferred=preferred,profiles=profiles,launches=launches)
(o/'result.json').write_text(json.dumps(out,indent=2)+'\n')
for shape,v in summary.items():
 print(shape,'control_us',v['control_us'],'launches',launches[shape])
 for n,z in v['ratios'].items():print(n,*(f'{z[k]:.6f}' for k in ('r','lo','hi','se')))
print('pairs',json.dumps(pairs));print('preferred',preferred)
