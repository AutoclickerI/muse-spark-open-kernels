"""Audit checksummed S19 stage pairs,resources and all point nominees;CI never vetoes."""
import argparse,ast,hashlib,json,lzma,base64,statistics,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(R/'tools'));import lab_packet
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args();host=json.loads(a.receipt.read_text());q=json.loads((D/'queue.json').read_text());payload=json.loads((D/'payload.json').read_text());manifest=json.loads((D/'manifest.json').read_text());cells=json.loads((D/'cells.json').read_text());counts=json.loads((D/'counts.json').read_text())['cells']
assert host['id']==q['job'];source=(R/q['source']).read_text();assert hashlib.sha256(source.encode()).hexdigest()==q['sha256'];entries=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='_LAB_PAYLOADS' for t in n.targets));name,sha,body=next(v for v in entries if v[1]==payload['sha256'][:16]);assert hashlib.sha256(body.encode()).hexdigest()==payload['sha256'];blob=next(n.args[0].value for n in ast.walk(ast.parse(body)) if isinstance(n,ast.Call) and getattr(n.func,'attr','')=='b85decode');expanded=lzma.decompress(base64.b85decode(blob)).decode();assert hashlib.sha256(expanded.encode()).hexdigest()==payload['expanded_sha256']
for spec in cells.values():
 text=(R/spec['cell']).read_text();assert hashlib.sha256(text.encode()).hexdigest()==spec['cell_sha256'] and text in expanded
packets=lab_packet.read_packets(host['result'].get('log_tail',''));record=next(v for v in packets if v['source_sha256']==sha);r=record['packet'];assert r['kind']=='P3A_INTPIPE' and r['complete'] and not r.get('transport_errors');assert r['identities']==payload['identities'];assert not r['checks']['failures'] and r['seeds']['completed']==r['seeds']['requested']==16;assert len(r['paired'])==128 and len(r['AA'])==64
for arm in ('BASE','AFFINE','PRIVATE'):assert r['checks']['gates'][arm][0]<=.015 and r['checks']['gates'][arm][1]<=.075
resources=[]
for m,arm,info in r['resources']:
 i=0 if m==1 else 1 if m==8 else 2;name=('BASE' if arm=='AFFINE' and m==1 else arm)+str(i);c=counts[name];smem=next(t['dynamic_smem'] for t in c['rows'] if t['M']==m and t['depth']==2);assert info[:2]==[2,c['registers']] and info[3:]==[smem,0],(m,arm,info,c);assert info[2] in (0,c['stack_bytes']),info;resources.append(dict(M=m,arm=arm,runtime=info,offline_match=True))
summary=[]
for m in (1,8,16,32):
 aa=[v for v in r['AA'] if v[1]==m];assert sorted(v[0] for v in aa)==list(range(16));aar=[v[4] for v in aa];abs_us=[abs(v[3]-v[2])/1000 for v in aa]
 for arm in ('AFFINE','PRIVATE'):
  pairs=[v for v in r['paired'] if v[1]==m and v[2]==arm];assert sorted(v[0] for v in pairs)==list(range(16));assert all(v[3]>0 and v[4]>0 and v[5]>0 for v in pairs);ratios=[v[5] for v in pairs];mean=statistics.mean(ratios);se=statistics.stdev(ratios)/4;alias=arm=='AFFINE' and m==1
  summary.append(dict(M=m,arm=arm,control='S19',ratio=mean,SE=se,CI95=[mean-2.131449545559323*se,mean+2.131449545559323*se],seeds=16,seed_indices=list(range(16)),seed_ratios=ratios,control_alias=alias,point_nominee=mean<1 and not alias,AA_mean_ratio=statistics.mean(aar),AA_SE=statistics.stdev(aar)/4,AA_mean_abs_us=statistics.mean(abs_us),AA_max_abs_us=max(abs_us)))
out=dict(job=q['job'],measured=True,host_passed=host['result'].get('passed'),invalid=[],source_audit=dict(verified=True,wrapper_sha256=q['sha256'],payload_sha256=payload['sha256'],expanded_sha256=payload['expanded_sha256'],packet_checksum=record['checksum'],exact_cells=len(cells)),resources=resources,summary=summary,seed_level_record=r,promotion='All valid point winners to P4,held-out six-load R lower bound>1 confirms;A:A/CI descriptive')
(D/'results.json').write_text(json.dumps(out,indent=2)+'\n');q.update(state='complete16seeds/TASK/AA/packet/source/resources verified',receipt=str(a.receipt.relative_to(R)) if a.receipt.is_absolute() else str(a.receipt));(D/'queue.json').write_text(json.dumps(q,indent=2)+'\n')
for v in summary:print(v['M'],v['arm'],round(v['ratio'],6),[round(t,6) for t in v['CI95']],v['point_nominee'],'AA',round(v['AA_mean_ratio'],6),v['AA_mean_abs_us'],v['AA_max_abs_us'])
