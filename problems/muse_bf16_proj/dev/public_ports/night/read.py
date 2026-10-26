"""Native whole-graph effects and per-site leave-outs, all rounds and seeds retained."""
import ast,base64,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
from promotion import interval
exec((H.parents[1]/'native_boundary/codec.py.inc').read_text())
def ci(x,aa=None):
 d=interval(x,aa);return [d['R'],*d['CI95']]
def analyse(receipt,wrapper,manifest):
 packets=read_packets(receipt['result']['log_tail']);assert packets and all(not p['packet'].get('transport_errors') and p['packet'].get('complete') is not False for p in packets)
 assert any(p['source_sha256']==manifest['payload_sha256'][:16] for p in packets)
 payloads=ast.literal_eval(next(n.value for n in ast.parse(wrapper).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_LAB_PAYLOADS' for t in n.targets)));assert any(hashlib.sha256(x[2].encode()).hexdigest()==manifest['payload_sha256'] for x in payloads)
 blob=next(s.split(' ',1)[1] for s in result_lines(receipt['result']['log_tail']) if s.startswith('NIGHT_GAP_XZ85 '));raw=nb_time_decode(blob);assert raw['complete'],raw['fatal'];assert raw['seeds']==manifest['seeds'];assert raw['parent']==manifest['selection']['parent'];assert raw['parent_points']==manifest['selection']['points'];assert {int(k):v for k,v in raw['parent_pool'].items()}=={int(k):v for k,v in manifest['selection']['pool'].items()}
 cells={tuple(r['case']):{a:[statistics.mean(v[a])*.008 for v in r['times']] for a in r['times'][0]} for r in raw['rows']};assert len(cells)==12
 for row in raw['rows']:assert len(row['times'])==8 and row['rounds']==2*len(row['times'][0]) and all(len(v)==row['rounds'] for z in row['times'] for v in z.values())
 out={}
 for older in ('INC','A','CODED_P32',*('NO_'+g for g in manifest['groups'])):
  per=[[] for _ in raw['seeds']];control=[[] for _ in raw['seeds']];loads={}
  for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
   sl,fu=cells[b,q,2048],cells[b,q,0];base=[46*x+16*y for x,y in zip(sl.get(older,sl['FULL']),fu.get(older,fu['FULL']))];new=[46*x+16*y for x,y in zip(sl['FULL'],fu['FULL'])];aa0=[46*x+16*y for x,y in zip(sl['A'],fu['A'])];aa1=[46*x+16*y for x,y in zip(sl['AA'],fu['AA'])];ratios=[a/n for a,n in zip(base,new)];aa=[a/n for a,n in zip(aa0,aa1)];loads[('d' if q==1 else 'v')+str(b)]=dict(R=ci(ratios,aa),saved_us=statistics.mean((a-n)/62 for a,n in zip(base,new)))
   for i in range(8):per[i].append(ratios[i]);control[i].append(aa[i])
  rr=[math.exp(statistics.mean(map(math.log,z))) for z in per];aa=[math.exp(statistics.mean(map(math.log,z))) for z in control];out[older]=dict(R=ci(rr,aa),loads=loads,per_seed_R=rr,per_seed_AA=aa)
 coded_vs_no=[a/b for a,b in zip(out['NO_P32']['per_seed_R'],out['CODED_P32']['per_seed_R'])];coded_interval=ci(coded_vs_no,out['NO_P32']['per_seed_AA'])
 return dict(job=receipt['id'],raw=raw,comparisons=out,coded_pool32_retention_R=coded_interval,coded_pool32_faster_than_raw=out['CODED_P32']['R'][0]<1,nominee_groups=[g for g in manifest['groups'] if out['NO_'+g]['R'][0]>1],limits='Whole-graph selection effects, not new promotion. INC/FULL includes selected parent portfolio; A/FULL isolates the new carrier portfolio. NO_site/FULL>1 favors retaining that site within FULL. Untouched leave-out shapes share exact FULL samples. All2N rounds used; max(primary,A:A)SE. CODED_P32/FULL is the whole coded-vs-raw T32 alternative; its other-site leave-outs are not measured. P4 confirms any selected composition with new held-out G0.')
if __name__=='__main__':
 p=Path(sys.argv[1])
 if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
 out=analyse(json.loads(p.read_text()),p.with_suffix('.py').read_text(),json.loads((H/'manifest.json').read_text()));(H/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v['R'] for k,v in out['comparisons'].items()},indent=2))
