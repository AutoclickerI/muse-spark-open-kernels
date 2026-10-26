"""Decode complete joint evidence and keep oracle/candidate claims separate."""
import base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
codec=dict(json=json,base64=base64,lzma=lzma);exec((H/'codec.py.inc').read_text(),codec)
def ci(values,log=True):
 z=[math.log(x) for x in values] if log else values;mu=statistics.mean(z);h=2.364624251*statistics.stdev(z)/math.sqrt(len(z));return [math.exp(x) for x in (mu,mu-h,mu+h)] if log else [mu,mu-h,mu+h]
def analyse(path):
 d=json.loads(path.read_text());tail=d['result'].get('log_tail','');lines=result_lines(tail);bs=[l.split(' ',1)[1] for l in lines if l.startswith('COMBINED_XZ85 ')]
 if not bs:
  out=dict(job=d['id'],valid=False,errors=[s[:2500] for s in lines if 'ERROR' in s or 'OVERSIZE' in s],reason='No joint archive; unmeasured');(H/'joint_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out);return out
 raw=codec['combined_decode'](bs[0]);transport=[e for p in read_packets(tail) for e in p['packet'].get('transport_errors',[])];oracle=[];candidate=[];primary=[];points=[]
 modes={(0,0,0):'RAW',(1,0,0):'X_smem',(0,1,0):'NO_MMA_occ',(0,0,1):'NO_EP',(1,1,0):'X_NO_MMA',(1,0,1):'X_NO_EP',(0,1,1):'NO_MMA_NO_EP',(1,1,1):'X_NO_MMA_NO_EP'}
 for cell in raw.get('oracle',{}).get('rows',[]):
  unit=raw['oracle']['unit_us'];v={a:[x*unit for x in vals] for a,vals in zip(cell['labels'],cell['producer'])};ratios={a:dict(vs_S19=ci([b/a0 for b,a0 in zip(values,v['S19'])]),vs_RAW=ci([b/a0 for b,a0 in zip(values,v['RAW'])]),us=ci(values,False)) for a,values in v.items()};split={}
  for axis,name in enumerate(('activation_TMA_barrier','tensor_issue_and_sync','epilogue')):
   samples=[]
   for seed in range(8):
    total=0
    for bits,label in modes.items():
     if bits[axis]:continue
     on=list(bits);on[axis]=1;size=sum(bits);weight=math.factorial(size)*math.factorial(2-size)/6
     total+=weight*(v[label][seed]-v[modes[tuple(on)]][seed])
    samples.append(total)
   split[name]=dict(us=ci(samples,False),samples=samples)
  split['residual']=dict(us=ci(v['X_NO_MMA_NO_EP'],False),samples=v['X_NO_MMA_NO_EP'])
  chains={a:[x*unit for x in vals] for a,vals in zip(cell.get('chain_labels',[]),cell.get('neighbour') or [])}
  chain_ratios={a:dict(vs_S19=ci([b/a0 for b,a0 in zip(values,chains['S19'])]),us=ci(values,False)) for a,values in chains.items()}
  oracle.append(dict(t=cell['t'],role=cell['role'],ratios=ratios,neighbour_ratios=chain_ratios,Shapley=split,resources=cell['occupancy']))
  print('ORACLE',cell['t'],cell['role'],{k:[round(x,4) for x in z['us']] for k,z in split.items()})
 for cell in raw.get('candidate',{}).get('rows',[]):
  unit=raw['candidate']['unit_us'];arms=cell['arms']
  for kind,key in (('producer','producer'),('neighbour','neighbour')):
   if cell.get(key) is None:continue
   vals=[[x*unit for x in a] for a in cell[key]];a=vals[0];rows=[]
   for j,arm in enumerate(arms[2:],2):
    row=dict(t=cell['t'],role=cell['role'],kind=kind,arm=arm,ratio=ci([b/a0 for b,a0 in zip(vals[j],a)]),AA=ci([b/a0 for b,a0 in zip(vals[1],a)]),A_us=statistics.mean(a),B_us=statistics.mean(vals[j]),samples=vals[j],control_samples=a);candidate.append(row);rows.append(row)
   if kind==('producer' if cell['role']=='qkv_gate' else 'neighbour'):
    row=min(rows,key=lambda r:r['ratio'][0]);primary.append(row);cfg=__import__('ss_source').CONFIGS[row['arm'][0]];points.append([row['t'],row['role'],cfg,row['arm'][1]]);print('BEST',row['t'],row['role'],cfg,row['arm'][1],row['ratio'])
 out=dict(job=d['id'],valid=raw.get('complete') and not transport,transport_errors=transport,receipt=str(path.relative_to(R)),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw=raw,oracle=oracle,candidate=candidate,primary=primary,CI='Student-t7 across8paired seeds; ratios log, saved-us differences arithmetic. Shapley allocates the three interacting conditional ablations; residual includes launch/prologue, initial X tiles and checksum overhead.')
 (H/'joint_results.json').write_text(json.dumps(out,indent=2)+'\n');(H/'ss_points.json').write_text(json.dumps(points,indent=2)+'\n');print('VALID',out['valid'],'FAILURES',raw.get('oracle',{}).get('failure'),raw.get('candidate',{}).get('failure'),raw.get('failure'));return out
if __name__=='__main__':
 path=Path(sys.argv[1])
 if not path.exists():path=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
 analyse(path.resolve())
