"""Whole-graph selection CIs against the frozen P4 S21 source, every round kept."""
import ast,base64,hashlib,json,lzma,math,statistics,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
from promotion import interval
exec((H.parent/'native_boundary/codec.py.inc').read_text())
def ci(x,aa=None):
 d=interval(x,aa);return [d['R'],*d['CI95']]
if __name__=='__main__':
 p=Path(sys.argv[1])
 if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
 receipt=json.loads(p.read_text());packets=read_packets(receipt['result']['log_tail']);assert packets and all(not p['packet'].get('transport_errors') and p['packet'].get('complete') is not False for p in packets)
 manifest=json.loads((H/'whole_manifest.json').read_text());assert any(p['source_sha256']==manifest['payload_sha256'][:16] for p in packets)
 wrapper=p.with_suffix('.py').read_text();tree=ast.parse(wrapper);payloads=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(getattr(a,'id','')=='_LAB_PAYLOADS' for a in n.targets)));assert any(hashlib.sha256(x[2].encode()).hexdigest()==manifest['payload_sha256'] for x in payloads)
 blob=next(s.split(' ',1)[1] for s in result_lines(receipt['result']['log_tail']) if s.startswith('PUBLIC_WHOLE_XZ85 '));raw=nb_time_decode(blob);assert raw['complete'],raw['fatal'];assert list(raw['seeds'])==manifest['seeds'] and raw['rounds']==manifest['rounds']
 cells={tuple(r['case']):{a:[statistics.mean(v[a])*.008 for v in r['times']] for a in r['times'][0]} for r in raw['rows']};assert len(cells)==12
 for r in raw['rows']:assert len(r['times'])==8 and all(len(z)==8 for v in r['times'] for z in v.values())
 arms={}
 for arm in (*manifest['points'],*manifest['pool']):
  if arm in manifest['pool'] and raw['pool_status'][arm]!='ready':arms[arm]=dict(measured=False,reason=raw['pool_status'][arm]);continue
  loads={};per_seed=[[] for _ in raw['seeds']];control=[[] for _ in raw['seeds']]
  for b,q in ((1,1),(16,1),(32,1),(1,8),(16,8),(32,8)):
   sl,fu=cells[b,q,2048],cells[b,q,0];a=[46*x+16*y for x,y in zip(sl['A'],fu['A'])];aa=[46*x+16*y for x,y in zip(sl['AA'],fu['AA'])];z=[46*x+16*y for x,y in zip(sl.get(arm,sl['A']),fu.get(arm,fu['A']))];rr=[x/y for x,y in zip(a,z)];ac=[x/y for x,y in zip(a,aa)]
   loads[('d' if q==1 else 'v')+str(b)]=dict(R=ci(rr,ac),time_ratio=ci([1/x for x in rr],ac),saved_us=statistics.mean((x-y)/62 for x,y in zip(a,z)),shapes={str(w):ci([x/y for x,y in zip(cells[b,q,w].get(arm,cells[b,q,w]['A']),cells[b,q,w]['A'])],[x/y for x,y in zip(cells[b,q,w]['AA'],cells[b,q,w]['A'])]) for w in (2048,0)})
   for i in range(8):per_seed[i].append(rr[i]);control[i].append(ac[i])
  rr=[math.exp(statistics.mean(map(math.log,z))) for z in per_seed];aa=[math.exp(statistics.mean(map(math.log,z))) for z in control];arms[arm]=dict(measured=True,R=ci(rr,aa),AA=ci(aa),loads=loads,points=manifest['points'].get(arm),per_seed_R=rr)
 out=dict(job=receipt['id'],base_sha256=manifest['base_sha256'],manifest=manifest,arms=arms,raw=raw,limits='Selection-stage six-load speedup with unchanged loads paired to same A. All8 fresh seeds, all8 rounds, larger primary/A:A SE. P4 needs new held-out G0 for any selected composition; no per-load veto.')
 (H/'whole_results.json').write_text(json.dumps(out,indent=2)+'\n')
 lines=['# Public projection and coded pool inside S21','',out['limits'],'',f"Receipt `{receipt['id']}`, S21 `{manifest['base_sha256'][:16]}`; {raw['checks']} layer +{raw['projection_checks']} strict projection/router TASK checks.",'','| Arm | Points | Six-load R [95% CI] | d16 us saved | d32 us saved |','|---|---|---|---:|---:|']
 for name,a in arms.items():
  if a['measured']:
   v,l,h=a['R'];lines.append(f"| {name} | {a.get('points') or 'pool 16/32'} | {v:.6f} [{l:.6f},{h:.6f}] | {a['loads']['d16']['saved_us']:.3f} | {a['loads']['d32']['saved_us']:.3f} |")
  else:lines.append('| '+name+' | unmeasured | '+a['reason']+' | | |')
 (H/'WHOLE.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
